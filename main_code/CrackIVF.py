from math import sqrt
import faiss
from vasili_helpers import *


class CrackIVF():
    def __init__(self, nlist=1, default_nprobe=20, niter=0, max_pts=256, seed=42, metric='euclidean', dbname=None, nthreads=None, min_pts=2, pts_crack_thr=64, alpha=0.5, cv_max=2, spread_max=10, size_prcnt_low=10, size_prcnt_high = 90):
        self.buffered_cracks = [] # list of cracks to be added
        self.converged = False
        self.running_assignments = None
        self.running_distances = None
        self.actual_assignments = None
        self.actual_distances = None
        self.running_cracks = None
        self.assignment_histogram = None # running assignments
        self.actual_assignment_histogram = None #  actual assignments
        self.MAX_CRACKS = 16_000  # upper limit recommendation from FAISS for <=1M datasets
        self.local_km_n_iter = 1 # just do 1 iterration of kmeans when refining

        self.index = None # the index 
        self.nlist = nlist  # number of centroids
        self.niter = niter  # number of global iterations at the start
        self.max_pts = max_pts # just use faiss default
        self.seed = seed
        self.verbose = False
        self.nprobe = default_nprobe
        self.N = None # total num points, we're in static dataset, set in .add()
        self.D = None # point dimensionality, set in .add()
        self.refine_nprobe = None # doens't matter we set it in refine TODO: set it in refine dynamically
        self.metric_str = metric # NOTE: needed in init_centroids <---- need to pass "dot" to avoid confusion w/ normalization
        if metric=='euclidean':
            self.metric = faiss.METRIC_L2
        elif metric =='angular' or metric=="dot":
            self.metric = faiss.METRIC_INNER_PRODUCT
        
        self.dbname = dbname
        self.nthreads = nthreads
        self.cost_estimator = CostEstimator(nthreads=self.nthreads)

        self.MIN_PTS_AFTER_CRACK = min_pts # NOTE: can keep it to 1 for final measurements for simplicity. 1 == you dont commit empty cracks (although 1 is quite low)
        self.THRESHOLD_PTS_PER_CENT = pts_crack_thr # better? for balance between sift/glove
        self.alpha = alpha
        self.cv_max = cv_max
        # self.spread_max = spread_max # RULE 2 OFF
        self.size_prcnt_low = size_prcnt_low
        self.size_prcnt_high = size_prcnt_high
        assert 100 - size_prcnt_high == size_prcnt_low # it's single parameter, just a bound
        self.is_initialized = False
        self.cracks = [] # keeps track of current cracks .reconstruct_n()
        
        self.total_search_ms = 0 # used cost estimator, track search times (ms)
        self.total_overhead_ms = 0 # used cost estimator, track overhead times (ms)
        
        #other metarada tracking
        self.total_pts_in_local_refine_region = [] # used to track how many points are in the local region of refine
        self.total_pts_percentage_in_local_refine_region = [] # -//- percentage -//-
        self.total_cracks_in_local_refine_region = [] # -//- cracks -//-
        self.total_batches_since_last_refine=0
        self.reorged_at_least_once = False
        self.total_past_queries = 0
        self.total_reorgs = 0 
        self.total_batches = 0 
        self.total_batches_since_last_crack = 0
        self.refines_since_last_crack = 0

        self.cum_dist_comp = 0

        self.rule_counts = {
            'rule_0_crack': 0, # same as rule_1 just checked elsewhere
            'rule_1_crack': 0, # same as rule_0 just checked elsewhere
            'rule_2_crack': 0,
            'rule_1_refine': 0,
            'rule_2_refine': 0,
            'rule_3_refine': 0,
            }
        
        print(f"CrackIVF initialized: {self.nlist=}, {self.niter=}, {self.max_pts=}, {self.seed=}, {self.metric_str=}, {self.MIN_PTS_AFTER_CRACK=}, {self.THRESHOLD_PTS_PER_CENT=}, {self.alpha=}, {self.cv_max=}, {self.size_prcnt_low=}, {self.size_prcnt_high=} | dbname={self.dbname} | nthreads={self.nthreads}")
        return 
    
    def add_dist_comp_model(self, n_centroids, n_points, n_iter=1, max_points=256, search_distances=False, comment=""):
        if not search_distances:
            P_train = min(n_points, n_centroids * max_points)
        elif search_distances:
            # assert 'train' not in comment, "train requires different distance computations"
            P_train = n_points
        self.add_dist_comp(n_centroids * P_train * n_iter, comment)

    def add_dist_comp(self, ncomp, comment=""):
        if len(comment) > 0:
            print(f"Adding {ncomp} distance computations for {comment}")
        self.cum_dist_comp += ncomp

    def add(self, xb):
        self.N, self.D = xb.shape
        print(f"Init Index... {self.nlist=} {self.niter=} {self.max_pts=} {self.seed=} {self.D=}")
        quantizer = faiss.IndexFlat(self.D, self.metric)
        self.index = faiss.IndexIVFFlat(quantizer, self.D, self.nlist, self.metric)
        self.index.cp.seed = self.seed
        self.index.cp.niter = self.niter
        self.index.cp.max_points_per_centroid = self.max_pts
        self.index.cp.nredo = 1
        self.index.verbose = self.verbose

        strain = time.perf_counter()
        self.index.train(xb)
        etrain = time.perf_counter()

        self.add_dist_comp_model(n_centroids=self.index.nlist, n_points=self.N, n_iter=self.niter, max_points=self.max_pts,search_distances=False, comment="index.train")
        
        ivf_stats = faiss.cvar.indexIVF_stats
        ivf_stats.reset()
        sadd = time.perf_counter()
        d, a = self.index.add_and_search(xb)
        if self.niter == 0: _ = init_centroids_after_assignments(self.index, metric=self.metric_str) # if no training, at least update centroids based on assignments
        self.N, self.D = xb.shape[0], xb.shape[1] # NOTE: Needed in cost estimator
        self.add_dist_comp(ivf_stats.ndis, "index.add_and_search") # distance comps from searching to add 
        assert ivf_stats.ndis == 0

        self.running_distances, self.running_assignments = d.reshape(-1), a.reshape(-1)
        self.actual_distances, self.actual_assignments = self.running_distances.copy(), self.running_assignments.copy() # NOTE: needed to avoid commiting bad cracks, & keep track of actual state  #TODO: figure out how to avoid it

        self.running_cracks = np.ones(self.MAX_CRACKS, dtype=int).reshape(-1)  # TODO: should be running_crack_count or smth, "running_cracks" is confusing
        self.assignment_histogram = np.bincount(self.running_assignments.flatten(), minlength=self.MAX_CRACKS) # initial assignments
        
        self.actual_assignment_histogram = get_actual_assignment_histogram(self.index, self.MAX_CRACKS)
        self.threshold_cluster_size = np.mean(self.assignment_histogram)
        self.large_clusters = np.where(self.assignment_histogram > self.threshold_cluster_size)[0]
        
        # NOTE: eadd - sadd includes all of the histogram overhead etc.
        eadd = time.perf_counter()

        print(f"\t---> Index Train Time = {(etrain - strain)*1000} ms | Add Time = {(eadd - sadd)*1000} ms<---")
        self.cracks = self.index.quantizer.reconstruct_n()
        self.is_initialized = True
        self.ivf_stats = faiss.cvar.indexIVF_stats

        # keep track of original
        # NOTE: not necessary anywhere in the index code, maybe i use it externally for testing I don't remember
        self.original_index = faiss.clone_index(self.index)
        self.original_cracks = self.cracks.copy()
        
        self.total_overhead_ms += (etrain - strain) * 1000
        self.total_overhead_ms += (eadd - sadd) * 1000
        
        return etrain - strain, eadd - sadd
    

    def search(self, queries, k=100):
        DEBUG=False
        if DEBUG: self.total_bad_buffered_cracks = 0 # TODO for debuging the refine bug 
        timings = {
                "search": [],
                "crack": [],
                "get_local_region": [],
                "metadata_tracking": [],
                "refine": [],
                "reorg": [],
                "total": [],
                "inner_point_assignments":[],
                "update_invlists":[],
                "remove_ids":[],
            }
        
        ### 1) search
        if not self.converged:
            self.nprobe = self.get_dynamic_nprobe() 
            
        extra_k = 1_000
        if self.converged: 
            extra_k=k
        
        search_start = time.perf_counter()  
        D, I, cracks_visited, _ = self.__search(queries, self.nprobe, extra_k)
        search_end = time.perf_counter()
        timings["search"].append(search_end - search_start)

        # for tracking progress
        if self.total_batches % 100 == 0 and not self.converged:
            # to explain cum distance differences, print median pts and cracks:
            median_pts = np.median(self.total_pts_in_local_refine_region) if self.total_pts_in_local_refine_region else 0
            median_cracks = np.median(self.total_cracks_in_local_refine_region) if self.total_cracks_in_local_refine_region else 0
            median_percentage = np.median(self.total_pts_percentage_in_local_refine_region) if self.total_pts_percentage_in_local_refine_region else 0
            pts_percentile_90 = np.percentile(self.total_pts_percentage_in_local_refine_region, 90) if self.total_pts_percentage_in_local_refine_region else 0
            print(f" ======> qid={self.total_batches*len(queries)} @ {self.total_batches=}) - {self.index.nlist=} - |C_buffered| = {len(self.buffered_cracks)}  - {self.total_search_ms=} {self.total_overhead_ms=} - \n{self.rule_counts.items()=} || median_pts_in_local_refine_region={median_pts} - median_cracks_in_local_refine_region={round(median_cracks,2)} - median_percentage_in_local_refine_region={round(median_percentage,2)}% - 90th_percentile_pts_ratio={round(pts_percentile_90,2)}%<====")
        
        
        qid = self.total_batches*len(queries)
        # # TODO: automate the crack rate monitoring and convergence when no new cracks and refines on same region to no benefit
        if qid > 200_000:
            self.converged = True
            # clear all state, should not be needed anymore, you only need self.index
            self.cracks = None
            self.buffered_cracks
            self.running_assignments = None
            self.running_distances = None
            self.actual_assignments = None
            self.actual_distances = None
            self.running_cracks = None
            self.assignment_histogram = None
            self.actual_assignment_histogram = None
            self.cost_estimator = None

        if self.converged:
            timings["refine"] = [0]
            total_end = time.perf_counter()
            timings["total"].append(total_end - search_start)
            # self.global_timings ADD timings # TODO unused for now 
            self.total_search_ms += (search_end - search_start) * 1000 # search
            self.total_overhead_ms += (total_end - search_end) * 1000 # overhead
            return D[:,:k], I[:,:k], timings
        

        metadata_start = time.perf_counter()
        self.total_past_queries += queries.shape[0]
        self.total_batches += 1
        self.total_batches_since_last_crack += 1
        self.total_batches_since_last_refine += 1

        if self.total_batches_since_last_crack == 1:
            self.index.invlists.print_stats()

        for qoffs, crack_candidate in enumerate(queries):
            cracks_query_visited = cracks_visited[qoffs:qoffs+1,:].ravel() # in sorted by distance order!

            GOOD_CRACK_CANDIDATE = True
            if GOOD_CRACK_CANDIDATE and len(self.buffered_cracks) + self.index.nlist < self.MAX_CRACKS :# and len(self.buffered_cracks) + self.index.nlist < self.MAX_CRACKS :
                new_crack_id = self.index.nlist + len(self.buffered_cracks) # len(.) starts from 0 before we add a crack so it will get crack_id = self.index.nlist
                
                indices = I[qoffs]
                mask_index = indices != -1
                indices = indices[mask_index] # drop -1 if in there
                distances = D[qoffs]
                distances = distances[mask_index] # drop -1 if in there
                if self.metric_str == "euclidean":
                    # euclidean: lower is better
                    distances_mask = distances < self.running_distances[indices].flatten() 
                elif self.metric_str in ["angular","dot"]:
                    # angular/dot: higher is better
                    distances_mask = distances > self.running_distances[indices].flatten() 
                final_mask = distances_mask 
                rule_0_crack = sum(distances_mask) <= self.MIN_PTS_AFTER_CRACK
                if rule_0_crack:
                    GOOD_CRACK_CANDIDATE = False
                    self.rule_counts['rule_0_crack'] += 1
                    continue # continue to evaluate next query, don't break and skip batch

                closer_to_new_crack_indices = indices[distances_mask]
                
                if self.MIN_PTS_AFTER_CRACK > 0:
                    # what are the current assignments of the points we're about to steal
                    cur_cents_of_visited_pts = self.running_assignments[closer_to_new_crack_indices].flatten()
                    # how many points are we about to steal from each centroids
                    affected_centroids, removal_counts = np.unique(cur_cents_of_visited_pts, return_counts=True)
                    # how many will be left in each centroid after 
                    remaining_points = self.assignment_histogram[affected_centroids] - removal_counts
                    # which centroids will be left with less than self.MIN_PTS_AFTER_CRACK
                    below_threshold_centroids = affected_centroids[remaining_points < self.MIN_PTS_AFTER_CRACK]

                    if any(below_threshold_centroids):
                        below_threshold_set = set(below_threshold_centroids)
                        mask_of_cents_more_than_threshold = np.isin(self.running_assignments[indices].flatten(), list(below_threshold_set), invert=True)
                        final_mask = final_mask & mask_of_cents_more_than_threshold  # TODO: AND NO POINTS IN BELOW THREHOLD SET CENTROIDS


                point_indices_to_steal = indices[final_mask]
                # NOTE: prev_assignments comes from running_assignments <--- this includes buffered cracks, so we don't need the argsort from Bowen
                prev_assignments = self.running_assignments[point_indices_to_steal].flatten()

                total_pts_in_region = np.sum(self.actual_assignment_histogram[cracks_query_visited])
                total_cracks_in_region = 1 + np.sum(self.running_cracks[cracks_query_visited]) # this also adds cracks to nearest centroid
                pts_to_crack_ratio = total_pts_in_region / total_cracks_in_region

                
                rule_1_crack = len(point_indices_to_steal) <= self.MIN_PTS_AFTER_CRACK # rule_1 same as rule_0, dont steal if <= MIN_PTS
                rule_2_crack = pts_to_crack_ratio < self.THRESHOLD_PTS_PER_CENT

                crack_rules = {
                    'rule_1_crack': len(point_indices_to_steal) <= self.MIN_PTS_AFTER_CRACK, # this is === rule_0, so we should not differentiate
                    'rule_2_crack': pts_to_crack_ratio < self.THRESHOLD_PTS_PER_CENT, # this kicks in a lot after some point, indicate "there are too many centroids"
                }

                # if rule_1_crack or rule_2_crack or rule_2_crack:
                if rule_1_crack or rule_2_crack: # keep only (rule 0/1 which are the same) and rule 3, for simplicity 
                    GOOD_CRACK_CANDIDATE = False 
                    for rule, triggered in crack_rules.items():
                        if triggered:
                            self.rule_counts[rule] += 1
                    break
                    continue # continue to evaluate next query, don't break and skip batch 
                # if  (len(point_indices_to_steal) <= self.MIN_PTS_AFTER_CRACK) or (not affects_a_large_cluster) or (pts_to_crack_ratio < self.THRESHOLD_PTS_PER_CENT):
                
                # TODO BUG???: Make sure this histogram is updated correctly...
                self.running_distances[point_indices_to_steal] = distances[final_mask]
                self.running_assignments[point_indices_to_steal] = new_crack_id
                
                # NOTE: if cracks_query_visited is actual crack then this is actually nearest ( from current index )
                # TODO AT REFACTO: change the naming in the codebase so that it's clear. actual_ can mean current index, running_  or dyn_ can mean anything dynamically changing uncommited
                #    - self.running_cracks should be dyn_local_crack_count or something, it keeps counts, not ids of cracks
                nearest_crack = cracks_query_visited[0]
                self.running_cracks[nearest_crack] += 1
                
                # histogram update
                np.subtract.at(self.assignment_histogram, prev_assignments, 1)  # Decrease count of previous centroids
                np.add.at(self.assignment_histogram, new_crack_id, len(point_indices_to_steal))  # Increase count for new centroid
                
                # update mean and large clusters
                self.threshold_cluster_size = np.mean(self.assignment_histogram)
                self.large_clusters = np.where(self.assignment_histogram > self.threshold_cluster_size)[0]
            
                self.buffered_cracks.append(crack_candidate)

        
        ## NOTE: This is the Cost Estimator for REORG
        N_crack_locs, Nc = len(self.buffered_cracks), self.index.nlist
        N_vis = self.index.ntotal # If you want to be "pessimistic" and assume entire dataset in visited

        # size_condition is a heuristic to avoid going from 100 partitions to 101, where the improvement would be minimal
        #       - we want to commit cracks when enough of "GOOD_CRACK_CANDIDATES"  have gathered s.t. we expect response time improvement
        #       - only case to bypass crack condition, is if we're about to reach the max_cracks (but maybe we can even remove this for simplicity)
        # NOTE: we can say there's a sweet spot of buffered cracks where cracking will take priority, we can say on our experiments it's 20-40% of current size
        size_condition = (N_crack_locs > int(0.2*Nc) ) or N_crack_locs + Nc == self.MAX_CRACKS
        # size_condition = N_crack_locs > 1000 or N_crack_locs + Nc == self.MAX_CRACKS
        COMMIT_REORG = False
        CAN_AFFORD_REORG = self.cost_estimator.is_there_enough_budget_for_reorg(self.alpha, self.total_search_ms, self.total_overhead_ms, N_crack_locs, Nc, N_vis, self.N, self.D, self.MAX_CRACKS)
        if CAN_AFFORD_REORG and size_condition and self.index.nlist < self.MAX_CRACKS:
            COMMIT_REORG = True

        metadata_end = time.perf_counter()
        timings["metadata_tracking"].append(metadata_end - metadata_start) # TODO: fix timing functions
        
        ### 3) REFINE 
        total_refine_time = 0
        total_get_local_region_time = 0
        start_refine = time.perf_counter()

        if not COMMIT_REORG and self.reorged_at_least_once:
            N_crack_locs, Nc = 1, self.index.nlist
            C_vis = self.nprobe
            N_vis = int(0.05 * self.index.ntotal) # NOTE: we can get this from histogram easy
            for i, query in enumerate(queries):
                GOOD_REFINE_CANDIDATE=False
                REFINE_AROUND_QUERY=False

                if not self.cost_estimator.is_there_enough_budget_for_refine(self.alpha, self.total_search_ms, self.total_overhead_ms, N_crack_locs, Nc, N_vis, C_vis, self.D, self.max_pts, self.local_km_n_iter):
                    break 
                
                query = query.reshape(1,-1)
                cracks_query_visited = cracks_visited[i:i+1,:].ravel()

                hist_values = self.actual_assignment_histogram[cracks_query_visited] # get imbalance with all nearby centroids, not only affected ones
                CV_local_imbalance = np.std(hist_values) / (np.mean(hist_values) + 1e-8)  # std/mean is one type of variation 

                rule_1_refine = CV_local_imbalance > self.cv_max

                small_cracks_in_region = cracks_query_visited[self.actual_assignment_histogram[cracks_query_visited] <= self.threshold_small_cluster_current_index]
                large_cracks_in_region = cracks_query_visited[self.actual_assignment_histogram[cracks_query_visited] >= self.threshold_large_cluster_current_index] # NOTE: identify if a large crack in local region
                rule_2_refine = len(small_cracks_in_region) > 0 and len(large_cracks_in_region) > 0

                refine_rules = {
                        'rule_1_refine': CV_local_imbalance > self.cv_max,
                        # 'rule_2_refine': local_spread > self.spread_max, # RULE 2 OFF
                        'rule_2_refine': len(small_cracks_in_region) > 0 and len(large_cracks_in_region) > 0, # NOTE: good for glove, but on coco it refine too much
                    }

                if (rule_1_refine or rule_2_refine):
                    for rule, triggered in refine_rules.items():
                        if triggered:
                            self.rule_counts[rule] += 1
                    GOOD_REFINE_CANDIDATE = True # unused just conceptual
                    REFINE_AROUND_QUERY = True
 
                    self.refine_nprobe = self.nprobe 
                start_get_local_region = time.perf_counter()

                #### 2) REFINE 
                if REFINE_AROUND_QUERY and self.reorged_at_least_once:
                    self.total_batches_since_last_refine = 0 
                    print(F"qid={self.total_batches*len(queries)} cracks={self.index.nlist} - IN REFINE_AROUND_QUERY ")
                    
                    c_visited_ids, c_visited, pids_visited, p_visited = self.__get_local_region(query, self.refine_nprobe)
                  
                    
                    total_get_local_region_time += (time.perf_counter() - start_get_local_region)
                    # TODO: keep track of total pounts and total cracks in this local refine
                    self.total_pts_in_local_refine_region.append(p_visited.shape[0])
                    self.total_pts_percentage_in_local_refine_region.append(round((p_visited.shape[0]/self.index.ntotal)*100,2))
                    self.total_cracks_in_local_refine_region.append(c_visited.shape[0])

                    print(f"{round((p_visited.shape[0]/self.index.ntotal)*100,2)} % of total pts : {c_visited_ids.shape=}, {c_visited.shape=}, {pids_visited.shape=}, {p_visited.shape=}")
                    local_quantizer = self.__refine_cracks_locally(c_visited_ids, c_visited, p_visited)
                    time_assign, time_add, time_remove = self.__commit_reorg(local_quantizer, 
                                                                            c_visited_ids, 
                                                                            pids_visited, 
                                                                            p_visited, 
                                                                            use_running_assignments=False, #don't use True here
                                                                            )
                    self.threshold_small_cluster_current_index = np.percentile(self.actual_assignment_histogram, self.size_prcnt_low)
                    self.threshold_large_cluster_current_index = np.percentile(self.actual_assignment_histogram, self.size_prcnt_high)
                    print(f"-- AFTER refine {self.index.nlist} : |C_buffered|= {len(self.buffered_cracks)} {self.index.invlists.compute_ntotal()=}={self.index.ntotal=} -- Imbalance factor = {self.index.invlists.imbalance_factor()=}")

        total_refine_time += (time.perf_counter() - start_refine)

        if COMMIT_REORG and len(self.buffered_cracks) > 0:
            HAVE_STUFF_WE_WANT_TO_COMMIT = True # NOTE: this changes if you remove all buffered cracks
            self.reorged_at_least_once = True
            crack_start = time.perf_counter()
            assert len(self.buffered_cracks) > 0
            self.total_reorgs += 1
            self.total_batches_since_last_crack = 0
            self.refines_since_last_crack = 0
 
            # 0. state
            if DEBUG:
                print(f"{len(self.buffered_cracks)=}") # python list of (1,d) buffered cracks 
                print(f"{self.buffered_cracks[0].shape=}") # python list
                print(f"{self.running_assignments.shape=}") # (1,N) : dynamic assignments, includes buffered crack ids
                print(f"{self.running_distances.shape=}") #  (1,N) : dynamic distances, includes distances to buffered crack ids
                print(f"{self.running_cracks.shape=}") # (1,N): TODO: needed?, keeps track of cracks 
                print(f"{self.assignment_histogram.shape=}")
                print(f"{self.actual_assignment_histogram.shape=}")
                print(f"{self.cracks.shape=}") # commited cracks, or just about to be commited

            # Slice the buffered cracks part of the histogram 
            buffered_assign_hist = self.assignment_histogram[self.index.nlist : self.index.nlist + len(self.buffered_cracks) ]
            # Get indices where the count is below MIN_PTS_AFTER_CRACK
            bad_buffered_cracks = np.where(buffered_assign_hist < self.MIN_PTS_AFTER_CRACK)[0] + self.index.nlist # NOTE: you need to add .nlist to correct crack_ids, since sliced histogram starts from 0
            prev_nlist = self.index.nlist
            ts = time.perf_counter()
            # NOTE: I think, refine can still create cracks with fewer poitns thatn MIN_PTS_AFTER_CRACK, but at least we wont commit a buffered crack with that amount <-- 
            self.__remove_buffered_crack_corrected(bad_buffered_cracks)
            te = time.perf_counter()
            print(f"---- __remove_buffered_crack took {(te-ts)*1000} ms for {bad_buffered_cracks.shape[0]} bad_buffered_cracks")
            if len(self.buffered_cracks) == 0:
                # removed all buffered cracks
                assert prev_nlist == self.index.nlist # didn't change/commit anything
                HAVE_STUFF_WE_WANT_TO_COMMIT = False

            if DEBUG:
                self.total_bad_buffered_cracks = len(bad_buffered_cracks)
                self.bad_buffered_cracks = bad_buffered_cracks
                if DEBUG and self.total_bad_buffered_cracks > 0:
                    total_pts_in_bad_cracks = buffered_assign_hist[np.where(buffered_assign_hist < self.MIN_PTS_AFTER_CRACK)].sum()
                    self.total_pts_in_bad_cracks = self.assignment_histogram[bad_buffered_cracks].sum()
                    total_pts_about_to_be_reassigned = buffered_assign_hist.sum()
                    print(f" -----> {self.total_bad_buffered_cracks=} {total_pts_in_bad_cracks=} - {total_pts_about_to_be_reassigned=} --- {round((total_pts_in_bad_cracks/(total_pts_about_to_be_reassigned))*100,2)} % of pts about to be reorged are on bad cracks <------")

                    ts = time.perf_counter()

                    self.__remove_buffered_crack_corrected(bad_buffered_cracks)

                    te = time.perf_counter()
                    print(f"---- __remove_buffered_crack took {(te-ts)*1000} ms")



            # - CRACK + commit crack immediately:
            if DEBUG and self.total_bad_buffered_cracks > 0: print(f"BEFORE COMMIT_CRACKS") # FOR DEBUG
            if HAVE_STUFF_WE_WANT_TO_COMMIT:
                prev_nlist = self.index.nlist # NOTE: we already computed it further up
                new_cracks = self.__commit_cracks() # does not assign points, should be atomic
                assert self.index.nlist == prev_nlist + new_cracks.shape[0]
                crack_end = time.perf_counter()
                timings["crack"].append(crack_end - crack_start)
                ##### ^^^^ UP TO HERE ^^^^^: you added invlists for each crack, updated the quantizer, but have not assigned points to the cracks
                # get local region around the new cracks
                
                print(f"qid={self.total_batches*len(queries)} cracks={self.index.nlist} - IN COMMIT_REORG")
                start_get_local_region = time.perf_counter()
                if DEBUG and self.total_bad_buffered_cracks > 0: print(f"BEFORE_LOCAL_PARTITIONS") # FOR DEBUG

                local_partitions = np.where(self.actual_assignment_histogram != self.assignment_histogram)[0]
                # we want to grab points in get_local_region, there are no points in buffered cracks only grab from existing cracks
                local_partitions = local_partitions[local_partitions < self.index.nlist]
                if DEBUG and self.total_bad_buffered_cracks > 0: print(f"BEFORE GET LOCAL REGION") # FOR DEBUG
                c_visited_ids, c_visited, pids_visited, p_visited = self.__get_local_region(new_cracks, self.nprobe, local_partitions) # has copies, should be a faster way
                if DEBUG and self.total_bad_buffered_cracks > 0: print(f"AFTER GET LOCAL REGION") # FOR DEBUG
                total_get_local_region_time += (time.perf_counter() - start_get_local_region)
            
                local_quantizer = faiss.IndexFlat(self.D, self.metric)
                local_quantizer.add(c_visited)
                if DEBUG and self.total_bad_buffered_cracks > 0: print(f"BEFORE COMMIT REORG") # FOR DEBUG
                commit_reorg_start = time.perf_counter()
                time_assign, time_add, time_remove = self.__commit_reorg(local_quantizer, 
                                                                        c_visited_ids, 
                                                                        pids_visited, 
                                                                        p_visited, 
                                                                        use_running_assignments=True,
                                                                        )
                
            
                if DEBUG and self.total_bad_buffered_cracks > 0: print(f"AFTER COMMIT REORG") # FOR DEBUG
                
                # BUG: something in here is using all cores
                self.cracks, d = update_centroids_and_distances(self.index, metric=self.metric_str, centroids=self.cracks, calculate_using_faiss=False)
                self.running_distances = d.reshape(-1) # TODO BUG: is this correct/helpful??
                self.actual_distances = self.running_distances.copy()
            
            self.assignment_histogram = self.actual_assignment_histogram.copy()
            self.threshold_small_cluster_current_index = np.percentile(self.actual_assignment_histogram, self.size_prcnt_low)
            self.threshold_large_cluster_current_index = np.percentile(self.actual_assignment_histogram, self.size_prcnt_high)
            if DEBUG:
                # TESTING
                np.testing.assert_array_equal(self.cracks, self.index.quantizer.reconstruct_n())
                np.testing.assert_array_equal(self.running_assignments, self.actual_assignments) #NOTE: TESTING
                np.testing.assert_array_equal(self.running_distances, self.actual_distances) #NOTE: TESTING

            self.running_cracks.fill(1) # reset all num cracks to 1 since we commited
            reorg_end = time.perf_counter()
            if not HAVE_STUFF_WE_WANT_TO_COMMIT:
                commit_reorg_start = time.perf_counter()
                total_get_local_region_time = (time.perf_counter() - commit_reorg_start) 
            timings["reorg"].append(reorg_end - commit_reorg_start)
            timings["inner_point_assignments"].append(time_assign)
            timings["update_invlists"].append(time_add)
            timings["remove_ids"].append(time_remove)
            
            assert len(self.buffered_cracks) == 0 # that it was reset

            assert self.index.nlist == self.index.invlists.nlist == self.index.quantizer.ntotal
            print(f"-- AFTER cracks {self.index.nlist} : {self.index.invlists.compute_ntotal()=}{self.index.ntotal=} -- Imbalance factor = {self.index.invlists.imbalance_factor()=}")

        timings["refine"].append(total_refine_time)
        timings["get_local_region"].append(total_get_local_region_time)
        total_end = time.perf_counter()
        timings["total"].append(total_end - search_start)

        self.total_search_ms += (search_end - search_start) * 1000 # search
        self.total_overhead_ms += (total_end - search_end) * 1000 # overhead
        
        return D[:,:k], I[:,:k], timings


    def __remove_buffered_crack_corrected(self, bad_buffered_cracks, DEBUG=False):
        if bad_buffered_cracks.shape[0] == 0: return 
        
        total_buffered_cracks_before = len(self.buffered_cracks)
        bad_buffered_cracks = np.sort(bad_buffered_cracks) # ensure sorted, although already is but whatever

        bad_cracks_in_buffer_positions = bad_buffered_cracks - self.index.nlist # so that you can iterate over buffered_cracks
        
        bad_buffer_positions = set(bad_cracks_in_buffer_positions) # for O(1) lookup

        old_to_new = {}

        bad_cracks = []
        for bad_crack_buffer_pos in bad_cracks_in_buffer_positions:
            bad_cracks.append(self.buffered_cracks[bad_crack_buffer_pos])

        # TODO: handlde the buffered_cracks array ( all at once before truncating... )
        last_good_buffer_pos = len(self.buffered_cracks) - 1
        # should be O(buffered cracks)
        for i, bad_crack_buffer_pos in enumerate(bad_cracks_in_buffer_positions):
            # find last good crack 
            while last_good_buffer_pos in bad_buffer_positions:# or last_good_buffer_pos <= bad_crack_buffer_pos:
                last_good_buffer_pos -= 1
                if last_good_buffer_pos < 0 or last_good_buffer_pos <= bad_crack_buffer_pos:
                    break 

            if last_good_buffer_pos <= bad_crack_buffer_pos:
                # since they're sorted...
                break

            last_good_crack_id = last_good_buffer_pos + self.index.nlist # actual crack id, the buffer pos is for self.buffered_cracks only
            new_crack_id = bad_buffered_cracks[i]

            old_to_new[last_good_crack_id] = new_crack_id

            self.buffered_cracks[bad_crack_buffer_pos], self.buffered_cracks[last_good_buffer_pos] = self.buffered_cracks[last_good_buffer_pos], self.buffered_cracks[bad_crack_buffer_pos]
            self.assignment_histogram[new_crack_id] = self.assignment_histogram[last_good_crack_id]
            self.assignment_histogram[last_good_crack_id] = 0 # to be removed
            
            last_good_buffer_pos -= 1

        
        last_positions = self.buffered_cracks[-len(bad_cracks):] if bad_cracks else []

        # Assert that all elements of `bad_cracks` are in the last `len(bad_cracks)` positions
        assert set(map(tuple, bad_cracks)) == set(map(tuple, last_positions)), \
            f"Assertion failed: bad_cracks not fully in last {len(bad_cracks)} positions: {len(last_positions)}"

        # Final remove from buffered_cracks
        truncated_len = len(self.buffered_cracks) - bad_buffered_cracks.shape[0]
        self.buffered_cracks = self.buffered_cracks[:truncated_len]
        # NOTE: The following is the slowest part of the function, the above is cheap: 
        

        # # -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----
        
        pids_to_be_reassigned = np.where(np.isin(self.running_assignments, bad_buffered_cracks))[0]
        self.running_assignments[pids_to_be_reassigned] = self.actual_assignments[pids_to_be_reassigned]
        self.running_distances[pids_to_be_reassigned] = self.actual_distances[pids_to_be_reassigned]
    
        
        max_id = self.running_assignments.max()
        if old_to_new:  # Check if old_to_new is not empty
            max_id = max(max_id, max(old_to_new.keys()))

        remap_array = np.arange(max_id + 1, dtype=self.running_assignments.dtype)
        for old, new in old_to_new.items():
            remap_array[old] = new
        new_assignments = remap_array[self.running_assignments]
        
        self.running_assignments[:] = new_assignments

        # # -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----# -----
        if DEBUG:
            print("...TESTING...")
            # CHEF NOTE : FOR TESTING ONLY UNCOMMENT:
            self.__test_remap_methods_equivelance(bad_buffered_cracks, old_to_new) # test loop vs mass remap
            assert pids_to_be_reassigned.shape[0] == self.total_pts_in_bad_cracks
            # print(f"{pids_to_be_reassigned.shape=} === {self.total_pts_in_bad_cracks}? <----- HERE\n")
            pids_that_should_not_exist = np.where(np.isin(self.actual_assignments, bad_buffered_cracks))[0]
            print(f"{pids_that_should_not_exist.shape=}")
            assert pids_that_should_not_exist.shape[0] == 0
            
            total_good_points_remapped = np.sum(self.running_assignments != new_assignments)
            repurposed_cracks = list(old_to_new.keys())
            # NOTE: assert no assignments left pointing to the "last crack ids" that where moved
            pids_should_not_exist = np.where(np.isin(self.running_assignments, np.array(repurposed_cracks)))[0]
            print(f"{pids_should_not_exist.shape=}")
            assert pids_should_not_exist.shape[0] == 0
            total_buffered_cracks_after = len(self.buffered_cracks)
            print(f"FINISHED - {len(bad_buffered_cracks)=} - {total_buffered_cracks_before=} {total_buffered_cracks_after=}" )

            # # self.__mass_replacement_remap()
            # self.__mass_replacement_remap(old_to_new) # avoid the above loop (B)
            
        return 


    def __test_remap_methods_equivelance(self, bad_buffered_cracks, old_to_new):
        # NOTE: DOES NOT ASSESS CORRECTNESS, ONLY EQUIVALENCE
        """
        Tests that the efficient remap assignments approach and the loop-based approach
        produce equivalent results for self.running_assignments and self.running_distances.
        """

        # Save copies of the original running arrays
        original_assignments = self.running_assignments.copy()
        original_distances   = self.running_distances.copy()

        # --- WAY 1: Efficient remap approach ---
        assignments_eff = original_assignments.copy()
        distances_eff   = original_distances.copy()

        # Restoration step: replace assignments for pids in bad_buffered_cracks
        pids_to_be_reassigned = np.where(np.isin(assignments_eff, bad_buffered_cracks))[0]
        assert pids_to_be_reassigned.shape[0] == self.total_pts_in_bad_cracks, (
            f"Expected {self.total_pts_in_bad_cracks} pts in bad cracks, got {pids_to_be_reassigned.shape[0]}"
        )
        pids_that_should_not_exist = np.where(np.isin(self.actual_assignments, bad_buffered_cracks))[0]
        print(f"pids_that_should_not_exist.shape = {pids_that_should_not_exist.shape}")
        assert pids_that_should_not_exist.shape[0] == 0, "Unexpected bad crack ids in actual assignments"
        
        assignments_eff[pids_to_be_reassigned] = self.actual_assignments[pids_to_be_reassigned]
        distances_eff[pids_to_be_reassigned]   = self.actual_distances[pids_to_be_reassigned]
        
        # Apply the efficient remap (vectorized) approach on assignments_eff.
        max_id = max(assignments_eff.max(), max(old_to_new.keys()))
        remap_array = np.arange(max_id + 1, dtype=assignments_eff.dtype)
        for old, new in old_to_new.items():
            remap_array[old] = new
        new_assignments_eff = remap_array[assignments_eff]
        total_good_points_remapped_eff = np.sum(assignments_eff != new_assignments_eff)
        assignments_eff[:] = new_assignments_eff  # update in place

        # --- WAY 2: Loop-based approach ---
        assignments_loop = original_assignments.copy()
        distances_loop   = original_distances.copy()

        # Restoration step: replace assignments for pids in bad_buffered_cracks
        pids_to_be_reassigned = np.where(np.isin(assignments_loop, bad_buffered_cracks))[0]
        assert pids_to_be_reassigned.shape[0] == self.total_pts_in_bad_cracks, (
            f"Expected {self.total_pts_in_bad_cracks} pts in bad cracks, got {pids_to_be_reassigned.shape[0]}"
        )
        pids_that_should_not_exist = np.where(np.isin(self.actual_assignments, bad_buffered_cracks))[0]
        print(f"pids_that_should_not_exist.shape = {pids_that_should_not_exist.shape}")
        assert pids_that_should_not_exist.shape[0] == 0, "Unexpected bad crack ids in actual assignments"
        
        assignments_loop[pids_to_be_reassigned] = self.actual_assignments[pids_to_be_reassigned]
        distances_loop[pids_to_be_reassigned]   = self.actual_distances[pids_to_be_reassigned]
        
        total_good_points_remapped_loop = 0
        repurposed_cracks_loop = []
        for old_crack_id, new_crack_id in old_to_new.items():
            repurposed_cracks_loop.append(old_crack_id)
            mask = (assignments_loop == old_crack_id)
            total_good_points_remapped_loop += np.sum(mask)
            assignments_loop[mask] = new_crack_id

        # --- Final assertions ---
        np.testing.assert_array_equal(assignments_eff, assignments_loop)
        np.testing.assert_array_equal(distances_eff, distances_loop)
        
        print("SUCCESS: Efficient remap approach matches the loop-based approach!")



    def __search(self, queries, nprobe, k):
        self.index.nprobe = nprobe
        return self.index.search_and_return_centroids(queries, k)

    def __get_local_region(self, crack_locations, refine_region_nprobe = 20, local_partitions=None):
        '''
        if local partitions is given no need to find them using the quantizer
            - they should only be given in REOG as of 19/02/25
        '''
        if local_partitions is None: 
            _, local_partitions = self.index.quantizer.search(crack_locations, refine_region_nprobe)
            self.add_dist_comp_model(n_centroids=self.index.nlist, n_points=crack_locations.shape[0], n_iter=1, max_points=0, search_distances=True, comment="__get_local_region.search")

        c_visited_ids = np.unique(local_partitions.ravel())
        c_visited_ids = c_visited_ids[c_visited_ids != -1]  # Remove -1 [ if present ]
        c_visited = self.cracks[c_visited_ids]
        # 3) get points in the region and reassign them
        pids_visited, p_visited = get_points_in_invlists(index=self.index, invlists=c_visited_ids, DEBUG=False)

        # TODO INVESTIGATE: NOTE: for lastfm-64-dot, it's float64 for some reason
        c_visited = c_visited.astype(np.float32)
        p_visited = p_visited.astype(np.float32)

        return c_visited_ids, c_visited, pids_visited, p_visited


    def __commit_cracks(self):
        for _ in self.buffered_cracks:
            faiss.downcast_InvertedLists(self.index.invlists).add_empty_list()
        # update nlist correctly
        self.index.nlist = self.index.invlists.nlist # change .nlist
        new_cracks = np.vstack(self.buffered_cracks) # convert to numpy array for easier search
        # get new quantizer ( not refined yet, + not replaced our current quantizer )
        # replace existing cracks with combined cracks and update the quantizer
        combined_cracks = np.vstack([self.cracks, new_cracks])
        new_global_quantizer = faiss.IndexFlat(self.D, self.metric)
        new_global_quantizer.add(combined_cracks)
        _ = replace_ivf_quantizer(self.index, new_global_quantizer)
        self.buffered_cracks = [] # reset buffered_cracks
        self.cracks = combined_cracks
        return new_cracks
    

    def __refine_cracks_locally(self, c_visited_ids, c_visited, p_visited):
        num_partitions = len(c_visited) # num partitions to refine/reorg
        local_km = faiss.Kmeans(self.D, num_partitions, niter=1, max_points_per_centroid=self.max_pts)
        local_km.verbose = self.verbose
        local_km.train(p_visited, init_centroids=c_visited) # initialize local quantizer and refine it
        self.add_dist_comp_model(n_centroids=num_partitions, n_points=p_visited.shape[0], n_iter=1, max_points=self.max_pts, search_distances=False, comment="__refine_cracks_locally.train")
        refined_local_quantizer = local_km.index # after it has been refined...
        
        self.cracks[c_visited_ids] = local_km.centroids # update the cracks NOTE: needed? I think I do use it somewhere important just forgot
        refined_global_quantizer = faiss.IndexFlat(self.D, self.metric)
        refined_global_quantizer.add(self.cracks) # after refinement            
        
        _ = replace_ivf_quantizer(self.index, refined_global_quantizer)
        return refined_local_quantizer

    def __commit_reorg(self, quantizer, c_visited_ids, pids_visited, p_visited, use_running_assignments=False, DEBUG=False):
        sass = time.perf_counter()
        went_to_crack_if_statment= False
        if use_running_assignments:
            went_to_crack_if_statment = True
            # CRACKS
            new_assignments = self.running_assignments[pids_visited]
            new_distances = self.running_distances[pids_visited]
        else:
            # REFINE 
            new_distances, new_assignments = quantizer.search(p_visited, k=1)
            n_centroids = quantizer.ntotal if hasattr(quantizer, 'ntotal') else quantizer.nlist
            self.add_dist_comp_model(n_centroids=n_centroids, n_points=p_visited.shape[0], n_iter=1, max_points=0, search_distances=True, comment="__commit_reorg.search")
            new_assignments = c_visited_ids[new_assignments.reshape(1,-1)] # make sure ids for centroids match 
            self.running_assignments[pids_visited] = new_assignments
            self.running_distances[pids_visited] = new_distances.reshape(-1)
            
        eass = time.perf_counter()
        if DEBUG and self.total_bad_buffered_cracks > 0: print(f"(IN __commit_reorg) - after if statements ") # FOR DEBUG
        ###### REMOVE
        sremove = time.perf_counter()
        for list_no in c_visited_ids:
            # NOTE: this clears the invlist, but it does not change outer_index.ntotal() correctly, we do it manually
            self.index.invlists.resize(int(list_no), 0) 
        self.index.ntotal -= len(p_visited) 
        eremove = time.perf_counter()
        if DEBUG and self.total_bad_buffered_cracks > 0: print(f"(IN __commit_reorg) - after resize all lists ") # FOR DEBUG
        # BUG: SEGFAULT (21/2/25)
        if DEBUG and self.total_bad_buffered_cracks > 0:
            assert went_to_crack_if_statment
            print(f"{self.bad_buffered_cracks.shape=}")
            print(f"{self.bad_buffered_cracks=}")
            print(f"{went_to_crack_if_statment=}")
            print(f"{self.index.nlist=}")
            print(f"{p_visited.shape=}")
            print(f"{pids_visited.shape=}")
            print(f"{new_assignments.shape=}")
            print(f"{new_assignments.shape=}")
            print(f"{self.running_assignments.max()=}")
            max_index = self.running_assignments.argmax()
            print(f"{self.bad_buffered_cracks.shape=}")
            print(f"{max_index in set(self.bad_buffered_cracks)}")
         
        ###### ADD BACK AGAIN (ntotal will be added)
        sadd = time.perf_counter()
        add_preassigned(index_ivf=self.index, x=p_visited, a=new_assignments.ravel(), ids=pids_visited.ravel()) 
        eadd = time.perf_counter()
        if DEBUG and self.total_bad_buffered_cracks > 0: print(f"(IN __commit_reorg) - after add preassigned all lists ") # FOR DEBUG
        self.assignment_histogram = np.bincount(self.running_assignments.flatten(), minlength=self.MAX_CRACKS) # running assingments changes with uncommited cracks
        
        
        self.actual_assignment_histogram = get_actual_assignment_histogram(self.index, self.MAX_CRACKS)
        

        if DEBUG:
            assert_actual_histogram_correct(self.index, self.MAX_CRACKS)
            crack_id_is_commited = (np.arange(len(self.assignment_histogram)) < self.index.nlist) 
            crack_ids = np.arange(len(self.assignment_histogram))  # Generate array of indices
            crack_id_is_buffered = (self.index.nlist <= crack_ids) & (crack_ids < (self.index.nlist + len(self.buffered_cracks)))
            crack_is_empty = (self.assignment_histogram == 0)
            almost_empty_crack = self.assignment_histogram < self.MIN_PTS_AFTER_CRACK
            commited_empty_crack_ids = np.where(crack_is_empty & crack_id_is_commited)[0]
            buffered_empty_crack_ids = np.where(crack_is_empty & crack_id_is_buffered)[0]
            bad_cracks_in_buffered_ids = np.where(almost_empty_crack & crack_id_is_buffered)[0]
            if len(commited_empty_crack_ids)>0 or len(buffered_empty_crack_ids)>0:
                print(f"\n ----> empty_commited= {commited_empty_crack_ids.shape=} & empty_in_buffered = {buffered_empty_crack_ids.shape} & bad_crack_in_buffered = {bad_cracks_in_buffered_ids.shape} & |C_buffered| = {len(self.buffered_cracks)} <----\n")
                self.index.invlists.print_stats()

        self.actual_assignments[pids_visited] = new_assignments
        self.actual_distances[pids_visited] = new_distances.reshape(-1)

        return (eass - sass), (eadd - sadd), (eremove - sremove)


    def get_dynamic_nprobe(self):
            nlists = [0,100,175,308,542,954,1676,2947,5179,9103,16000]
            match self.dbname:
                case "deep-image-96-angular":
                    nprobes = [2**2, 2**3, 2**3, 2**3, 2**4, 2**4, 2**4, 2**5, 2**5, 2**5, 2**6]
                case "glove-25-angular":
                    nprobes = [2**2, 2**3, 2**3, 2**4, 2**4, 2**5, 2**5, 2**5, 2**6, 2**6, 2**7] # "correct" from nprobe graphs notebook
                case "glove-50-angular":
                    nprobes = [2**3, 2**4, 2**5, 2**5, 2**6, 2**6, 2**7, 2**7, 2**7, 2**8, 2**8]
                case "glove-100-angular":
                    nprobes = [2**4, 2**5, 2**5, 2**6, 2**7, 2**7, 2**8, 2**8, 2**8, 2**9, 2**9]
                case "sift-128-euclidean" | "SIFT1M" | "SIFT10M":
                    nprobes = [5, 2**3, 2**3, 2**3, 2**4, 2**4, 2**5, 2**5, 2**6, 2**6, 2**6] 
                case "lastfm-64-dot":
                    nprobes = [16, 24, 34, 64, 64, 64, 64, 64, 64, 64, 64] # custom for 90-95+% recall

            for i in range(len(nprobes)):
                if nlists[i] < self.index.nlist <= nlists[i + 1]:
                    new_nprobe = min(nprobes[i], self.index.nlist)
                    if self.nprobe != new_nprobe:
                        print(f"nprobe_prev={self.nprobe} --> {new_nprobe=}")

                    self.nprobe = new_nprobe
                    return self.nprobe # in case <100 and nprobe too large
            return 0.1*self.index.nlist # if for some reason no match was found, ( eg >16k partitions, which I'm not looking into now)

class CostEstimator(): 
    def __init__(self, nthreads):
        self.nthreads = nthreads
        if self.nthreads != 16:
            print(f"WARNING: CostEstimator is not trained for {self.nthreads} threads. The estimated cost could be inaccurate.")

    def __reorg_cost_estimate_get_local_region(self, N_crack_locs, Nc, N_vis, D):
        X1, X2 = N_crack_locs * Nc, N_vis * D
        w1, w2, b = 9.792029439425269e-07, 8.848264329863606e-07, 3.587780694965886
        return w1*X1 + w2*X2 + b

    def __reorg_cost_estimate_commit_reorg_with_running_assignments(self, Nc, N_vis, D):
        X1, X2 = Nc, N_vis * D
        w1, w2, b = 4.257400629274415e-05, 3.3919302537771467e-07, 0.7444031045620599
        return w1*X1 + w2*X2 + b

    def __reorg_cost_estimate_init_centroids_after_assignment(self, N, D, Nc):
        X1, X2 = N*D, D*(N+Nc)
        w1, w2, b = -1.5951860751448968e-05, 1.7196139945158915e-05, 33.150811931567546
        return w1*X1 + w2*X2 + b

    def get_reorg_cost_estimate(self, N_crack_locs, Nc, N_vis, D, N):
        T_get_local_region = self.__reorg_cost_estimate_get_local_region(N_crack_locs, Nc, N_vis, D)
        T_commit_reorg = self.__reorg_cost_estimate_commit_reorg_with_running_assignments(Nc, N_vis, D)
        T_init_centroids_after_assignment = self.__reorg_cost_estimate_init_centroids_after_assignment(N, D, Nc) # overhead can be reduced
        return T_get_local_region + T_commit_reorg + T_init_centroids_after_assignment


    # TODO: fix this (IF NEEDED):
    def __refine_cost_estimate_get_local_region(self, N_crack_locs, Nc, N_vis, D):
        X1, X2 = N_crack_locs * Nc, N_vis * D
        # w1, w2, b = 9.792029439425269e-07, 8.848264329863606e-07, 3.587780694965886
        w1,w2, b = 0, 0, 0
        return w1*X1 + w2*X2 + b

    def __refine_cost_estimate_commit_reorg(self, Nc, N_vis, C_vis, D):
        X1, X2 = N_vis * C_vis * D + Nc, N_vis * D
        w1, w2, b = 8.912474562983087e-09, 1.2000867199316712e-06, -8.689369633586466
        return w1*X1 + w2*X2 + b

    def __refine_cost_estimate_local_kmeans(self, N_vis, max_pts, n_iter, C_vis, D, Nc):
        N_train = min(N_vis, max_pts * N_vis)
        X1, X2 = n_iter * N_train * C_vis * D, Nc * D
        w1, w2, b = 9.311884586956287e-09, 7.437262586898243e-05, 191.62760060405162
        return w1*X1 + w2*X2 + b

    def get_refine_cost_estimate(self, N_crack_locs, Nc, N_vis, D, max_pts, n_iter, C_vis):
        T_get_local_region = self.__refine_cost_estimate_get_local_region(N_crack_locs, Nc, N_vis, D) 
        T_local_kmeans = self.__refine_cost_estimate_local_kmeans(N_vis, max_pts, n_iter, C_vis, D, Nc)
        T_commit_reorg = self.__refine_cost_estimate_commit_reorg(Nc, N_vis, C_vis, D)
        return T_get_local_region + T_local_kmeans + T_commit_reorg

    def is_there_enough_budget_for_reorg(self, alpha, hist_search_ms, hist_overhead_ms, N_crack_locs, Nc, N_vis, N, D, max_cracks):
        reorg_cost_est_ms = self.get_reorg_cost_estimate(N_crack_locs, Nc, N_vis, D, N)
        
        total_est_after_op_ms = hist_search_ms + hist_overhead_ms + reorg_cost_est_ms
        total_build_est_after_op_ms = hist_overhead_ms + reorg_cost_est_ms
        if (total_build_est_after_op_ms <= alpha * total_est_after_op_ms):
            return True
        else:
            return False

    def is_there_enough_budget_for_refine(self, alpha, hist_search_ms, hist_overhead_ms, N_crack_locs, Nc, N_vis, C_vis, D, max_pts, n_iter):
        refine_cost_est_ms = self.get_refine_cost_estimate(N_crack_locs, Nc, N_vis, D, max_pts, n_iter, C_vis)

        total_est_after_op_ms = hist_search_ms + hist_overhead_ms  + refine_cost_est_ms
        total_build_est_after_op_ms = hist_overhead_ms + refine_cost_est_ms
        
        if total_build_est_after_op_ms <= alpha * total_est_after_op_ms:
            return True
        else: 
            return False