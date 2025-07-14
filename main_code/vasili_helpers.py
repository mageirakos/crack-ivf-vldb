# NOTE: Improt this by doing:  from vasili_helpers import *
import struct
import faiss
import numpy as np
import time
import random
import sys
import os 
from typing import Tuple
import h5py
import matplotlib.pyplot as plt
from faiss.contrib.inspect_tools import get_invlist, get_invlist_sizes
from faiss.contrib.ivf_tools import replace_ivf_quantizer, add_preassigned
from faiss.contrib.datasets import SyntheticDataset

username = os.environ.get('USER')

################################################# 
# Other
################################################# 

def get_faiss_opt_level():
    print(faiss.get_compile_options())

def get_faiss_install_location():
    faiss.__file__

# Function to create the folder if it doesn't exist
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")

################################################# 
# Creating Indexes & Cracking Code
################################################# 

def get_invlist_ids_only(invlists, l):
    ''' 
    returns only the list_ids of the inverted list.
    '''
    invlists = faiss.downcast_InvertedLists(invlists)
    ls = invlists.list_size(l)
    list_ids = np.zeros(ls, dtype='int64')
    ids = None
    try:
        ids = invlists.get_ids(l)
        if ls > 0:
            faiss.memcpy(faiss.swig_ptr(list_ids), ids, list_ids.nbytes)
    finally:
        if ids is not None:
            invlists.release_ids(l, ids)
    
    return list_ids

def get_invlist_ids(index, ls_of_invlists):
    '''
    ls_of_invlists: list of inverted lists to get pids from
    '''
    tmp_pids = []
    for invlist_id in ls_of_invlists:
        point_ids = get_invlist_ids_only(index.invlists, int(invlist_id)) 
        tmp_pids.append(point_ids)
    pids_visited = np.concatenate(tmp_pids, axis=0)
    return pids_visited


def assert_actual_histogram_correct(index, max_cracks):
    '''
    call this to assert actual histogram matches actual assignment function in vasili_helpers
    '''
    actual_assignments = get_actual_assignments(index)
    A = get_actual_assignment_histogram(index, max_cracks)
    B = np.bincount(actual_assignments.flatten(), minlength=max_cracks)
    assert np.array_equal(A, B), "PROBLEM: assignments from get_invlist() != assignments from invlists.list_size()"

def get_actual_assignment_histogram(index, max_cracks):
    '''
    It is important to get actual histogram this way. The other way had a BUG
    '''
    total_invlists = index.nlist
    actual_assignment_histogram = np.zeros((max_cracks,), dtype=int) # NOTE: initialize with all 0, up to max cracks
    # NOTE: then update actual histograms with what they are from list_size(id)
    for invlist_id in range(total_invlists):
        invlist_id = int(invlist_id)
        actual_assignment_histogram[invlist_id] = index.invlists.list_size(invlist_id)
    return actual_assignment_histogram


def get_actual_assignments(index):
    '''get the actual assignmets of points to inverted list of an index'''
    total_invlists = index.nlist
    N = index.ntotal
    actual_assignments = np.zeros((N,), dtype=int)
    for invlist_id in range(total_invlists):
        point_ids, codes = get_invlist(index.invlists, int(invlist_id)) 
        actual_assignments[point_ids] = int(invlist_id)
    actual_assignments.reshape(-1)
    return actual_assignments


def get_points_in_invlists(index, invlists, DEBUG=False):
    tmp_pids, tmp_codes = [], []
    for invlist_id in invlists:
        point_ids, codes = get_invlist(index.invlists, int(invlist_id)) 
        tmp_pids.append(point_ids)
        tmp_codes.append(codes)
        # if DEBUG:print(f"- In {invlist_id=} => {point_ids=}")
    pids_visited = np.concatenate(tmp_pids, axis=0)
    p_visited = np.concatenate(tmp_codes, axis=0).view("float32")
    return pids_visited, p_visited

#NOTE: this is to do single iteration of cracking
def search_and_crack_single_query(outer_index, q, nprobe=10, k=10, inner_km_niter=1, refine_nprobe=1, REFINE=True, DEBUG=False, metric="euclidean"):
    # BUG CHEF NOTE: there's a memory leak in add_preassigned
    # Initialize a dictionary to store times for each section
    timings = {
        "search": [],
        "get_c_visited": [],
        "get_p_visited": [],
        "inner_km_train": [],
        "inner_point_assignments": [],
        "replace_quantizer": [],
        "update_invlists": [],
        "crack": [],
    }

    # recalls, prcnt_pts_scanned = [], []
    orig_nprobe = nprobe
    refine_nprobe = refine_nprobe
    # refine_nprobe = 100

    d = outer_index.quantizer.d

    # 1. search
    # start_time = time.perf_counter()
    # q = q.reshape(1, -1)
    # NOTE: Dynamic nprobe
    
    # search with original nprobe, and train with factor*orig_nprobe
    outer_index.nprobe = orig_nprobe
    # print(f"{outer_index.nprobe=}")
    start_time = time.perf_counter() 
    D_p, I_p = outer_index.search(q, k) #TODO: see if I can get c visited, get p visited and assignemtns for q-means from .search() 
    end_time = time.perf_counter()
    timings["search"].append(end_time - start_time)
    
    ############### SEARCH FINISHED

    if not REFINE:
        return outer_index, D_p, I_p, timings, refine_nprobe
    
    # HACK to work around parallelism issue (the rest of the threads will have to wait )
    # NOTE: since we might have batch size, just pick the first query from the batch and crack/refine etc.
    # Would also work w/o it I think, but then it cracks the entire area of the batch? (all nprobe all centroids etc I think)
    if q.shape[0] > 1:
        q = q[0:1,:]


    # if not REFINE:
    #     return outer_index, D_p, I_p, timings, c_visited_ids
    
    t0_crack = time.perf_counter()
    # q_recall = compute_recall(I_p, gt_ids[qid,:].reshape(1,-1), k)
    # recalls.append(q_recall)

    # 1c. get c_visited
    # Query Loop
    outer_centroids = outer_index.quantizer.reconstruct_n() # or kmeans.centroids idk which is faster
    start_time = time.perf_counter()
    c_visited_ids = outer_index.quantizer.search(q, refine_nprobe)[1][0]
    c_visited = outer_centroids[c_visited_ids]
    end_time = time.perf_counter()
    timings["get_c_visited"].append(end_time - start_time)
    
    ##################################### REFINE:
    # 2a. get p_visited
    start_time = time.perf_counter()
    pids_visited, p_visited = get_points_in_invlists(index=outer_index, invlists=c_visited_ids, DEBUG=DEBUG)
    end_time = time.perf_counter()
    timings["get_p_visited"].append(end_time - start_time)
    
    # prcnt_pts_scanned.append(round(len(pids_visited)/outer_index.ntotal, 3))

    # 2c. initialize new inner_quantizer w/ c_visited centroids

    start_time = time.perf_counter()
    assert len(c_visited_ids) == refine_nprobe
    inner_km = faiss.Kmeans(d, refine_nprobe, niter=inner_km_niter, max_points_per_centroid=256) #NOTE:256 change max_points_per_centroids to know if you should subsample or not don't subsample inner km ( get inner pt assignments for free )
    inner_km.verbose = True
    # print(f"{c_visited.shape}")
    inner_km.train(p_visited, init_centroids=c_visited)
    end_time = time.perf_counter()
    timings["inner_km_train"].append(end_time - start_time)

    # inner_point_assignments
    start_time = time.perf_counter()
    inner_point_assignments = inner_km.index.assign(p_visited, k=1).reshape(1, -1)
    inner_point_assignments = c_visited_ids[inner_point_assignments]
    # prev_assignments = outer_index.quantizer.assign(p_visited, k=1).reshape(1, -1)
    end_time = time.perf_counter()
    timings["inner_point_assignments"].append(end_time - start_time)
    # NOTE: q might be a batch
    # print(f"--- HERE HERE HERE  {q.shape=} {inner_point_assignments.shape=} {c_visited.shape=}")
    

    # 2d. Replace quantizer
    start_time = time.perf_counter()
    outer_centroids[c_visited_ids] = inner_km.centroids
    if metric == "euclidean":
        new_quantizer = faiss.IndexFlatL2(d)
    elif metric == "angular":
        new_quantizer = faiss.IndexFlatIP(d)
    new_quantizer.add(outer_centroids)
    _ = replace_ivf_quantizer(outer_index, new_quantizer)
    end_time = time.perf_counter()
    timings["replace_quantizer"].append(end_time - start_time)

    start_time = time.perf_counter()
    outer_index.verbose = False # don't print
    # outer_index.remove_ids(pids_visited) # too slow
    # BUG: .ntotal() in outer_index is not updated when you reset like this. But the operation is much faster.
    for list_no in c_visited_ids:
            # BUG: this clears the invlist, but it does not change outer_index.ntotal() correctly
            outer_index.invlists.resize(int(list_no), 0) 
    add_preassigned(index_ivf=outer_index, x=p_visited, a=inner_point_assignments.ravel(), ids=pids_visited.ravel())
    end_time = time.perf_counter()
    outer_index.verbose = True

    timings["update_invlists"].append(end_time - start_time)

    t1_crack = time.perf_counter()
    timings["crack"].append(t1_crack - t0_crack)
    return outer_index, D_p, I_p, timings, refine_nprobe

def update_centroids_and_distances(index, debug=False, metric='euclidean', centroids=None, calculate_using_faiss=False):
    '''
    NOTE: numpy distance calcs much faster, to avoid copy & also sorting by distance <--- 
    I have testing code to assert they calculate the same thing, just need to change the code so that both calculated at the same time to test
    '''

    # Get all invlist indices and dimensionality
    invlists = np.arange(index.nlist)
    dims = index.d

    # Initialize centroids array if not provided
    if centroids is None:
        centroids = np.full((index.nlist, dims), np.nan, dtype=np.float32)
    
    # Allocate distances array (distance for each point in the index)
    distances = np.full(index.ntotal, np.nan, dtype=np.float32)

    for invlist_id in invlists:
        # Retrieve the point ids and the corresponding codes from the invlist
        point_ids, codes = get_invlist(index.invlists, int(invlist_id))

        # Reinterpret the codes as float32 points and reshape to (n_points, dims)
        points = codes.view("float32")

        # Calculate centroid for this invlist if it has points
        if points.size > 0:
            cent = np.mean(points, axis=0).reshape(1,-1)
            if metric == "angular" or metric == faiss.METRIC_INNER_PRODUCT:
                # NOTE: make sure we pass self.metric_str not self.metric to avoid "dot" falling into here
                cent = cent / np.linalg.norm(cent)
            centroids[invlist_id] = cent

        # # Compute distances from each point to the centroid:
        # if metric == 'euclidean' or metric == faiss.METRIC_L2:
        #     # L2 distance
        #     if calculate_using_faiss: 
        #         index_flat = faiss.IndexFlatL2(dims)
        #         index_flat.add(points)
        #         D_faiss, I = index_flat.search(cent, points.shape[0])
        #         mapped_back_ids = np.array(point_ids)[I]
        #         distances[mapped_back_ids] = D_faiss
        #     else:
        #         D_np = np.sum((points - cent)**2, axis=1)
        #         distances[point_ids] = D_np
        # elif metric in ['angular', 'dot'] or metric == faiss.METRIC_INNER_PRODUCT:
        #     # inner product
        #     if calculate_using_faiss: 
        #         index_flat = faiss.IndexFlatIP(dims)
        #         index_flat.add(points)
        #         D_faiss, I = index_flat.search(cent, points.shape[0])
        #     else:
        #         D_np = np.dot(points, cent.T).flatten()
        #         distances[point_ids] = D_np
        
        # Compute distances from each point to the centroid:
        if metric == 'euclidean' or metric == faiss.METRIC_L2:
            # L2 distance
            if calculate_using_faiss: 
                index_flat = faiss.IndexFlatL2(dims)
                index_flat.add(points)
                D_faiss, I = index_flat.search(cent, points.shape[0])
                # Flatten the outputs so they become 1D arrays.
                D_faiss = D_faiss.flatten()
                I = I.flatten()
                mapped_back_ids = np.array(point_ids)[I]
                distances[mapped_back_ids] = D_faiss
            else:
                D_np = np.sum((points - cent)**2, axis=1)
                distances[point_ids] = D_np
        elif metric in ['angular', 'dot'] or metric == faiss.METRIC_INNER_PRODUCT:
            # Inner product
            if calculate_using_faiss: 
                index_flat = faiss.IndexFlatIP(dims)
                index_flat.add(points)
                D_faiss, I = index_flat.search(cent, points.shape[0])
                D_faiss = D_faiss.flatten()
                I = I.flatten()
                mapped_back_ids = np.array(point_ids)[I]
                distances[mapped_back_ids] = D_faiss
            else:
                D_np = np.dot(points, cent.T).flatten()
                distances[point_ids] = D_np

        # Map FAISS indices back to original point ids
        DEBUG=False
        if DEBUG:
            # NOTE: assume you calculated D_faisss and D_np
            D_faiss = D_faiss.flatten()  # Now shape (n_points,)
            I = I.flatten()                  
            D_faiss_unsorted = np.empty_like(D_faiss)
            D_faiss_unsorted[I] = D_faiss
            if np.allclose(D_np, D_faiss_unsorted):
                print("The distances match!")
            else:
                print("Mismatch between D_np and D_faiss!")

        # debug = True
        if debug:
            print(f"In invlist {invlist_id}: point_ids = {point_ids}, "
                  f"num_points = {points.shape[0]}")

    # Replace the index’s quantizer with one built from the centroids.
    if metric == 'euclidean' or metric == faiss.METRIC_L2:
        new_quantizer = faiss.IndexFlatL2(dims)
    elif metric in ['angular', 'dot'] or metric == faiss.METRIC_INNER_PRODUCT:
        new_quantizer = faiss.IndexFlatIP(dims)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Filter out any invlists that did not contain points (and thus have NaN centroids)
    # valid_centroids = ~np.isnan(centroids).any(axis=1)
    # new_quantizer.add(centroids[valid_centroids])

    new_quantizer.add(centroids)
    replace_ivf_quantizer(index, new_quantizer)

    assert not np.isnan(distances).any(), "Distances array contains NaN values"
    assert distances.shape[0] == index.ntotal

    return centroids, distances

def init_centroids_after_assignments(index, debug=False, metric='euclidean', centroids=None):
    """
    Processes the inverted lists of the given index, computes centroids, and replaces the quantizer.

    Parameters:
    - index: The FAISS index object.
    - get_invlist: A function to retrieve point IDs and codes for an invlist ID.
    - replace_ivf_quantizer: A function to replace the quantizer of the FAISS index.
    - debug: If True, prints debug information.

    Returns:
    - pids_visited: Concatenated point IDs visited.
    - points_visited: Concatenated points visited.
    - centroids: Calculated centroids for each invlist.
    - elapsed_time: Time taken to replace the quantizer.
    """

    invlists = np.arange(index.nlist)
    dims = index.d  # Dimensionality of the vectors in the index
    # I need to initialize it with what it was not NaN...
    if centroids is None:
        centroids = np.full((len(invlists), dims), np.nan, dtype=np.float32)  # Initialize centroids with NaN
    
    
    for invlist_id in invlists:
        point_ids, codes = get_invlist(index.invlists, int(invlist_id))
        
        # Convert codes to float32 to treat as points
        points = codes.view("float32")
        
        # Calculate centroid for this invlist if it has points
        if points.size > 0:
            cent = np.mean(points, axis=0).reshape(1,-1)
            if metric == "angular" or metric == faiss.METRIC_INNER_PRODUCT:
                # NOTE IMPORTANT : ^^^^ this can create issues if metric = "dot" and we passed faiss.METRIC_INNER_PRODUCT
                #   - NOTE: so make sure you pass self.metric_str from CrackIVF to say "dot" not faiss.METRIC_INNER_PRODUCT
                cent = cent / np.linalg.norm(cent)
            centroids[invlist_id] = cent
            
        if debug:
            print(f"In {invlist_id=} => {point_ids=}, {points.shape=}")
        
    # Replace the quantizer
    if metric == "euclidean" or metric == faiss.METRIC_L2:
        new_quantizer = faiss.IndexFlatL2(dims)
    elif metric == "angular" or metric == faiss.METRIC_INNER_PRODUCT:
        print("normalized init_centroids_after_assignments")
        new_quantizer = faiss.IndexFlatIP(dims)
        # NOTE: I do it one centroid at a time above
        # faiss.normalize_L2(centroids) # NOTE: If you don't normalize, there's Recall instability, because refine has to fix centroids: check on angular dataset like glove-50-angular
    elif metric == "dot":
        new_quantizer = faiss.IndexFlatIP(dims)

    new_quantizer.add(centroids)
    replace_ivf_quantizer(index, new_quantizer)

    return centroids


def get_faiss_ivfflatL2_from_individual_components(data_pts, nlist=10, km_n_iter=10, km_max_pts=256, verbose=True):
    '''
    This creates and returns an IVFFlat index but built up from the individual components.
    Namely:
        - train kmeans for km_n_iter with km_max_pts subsampled from data_pts, to find centroids
        - create IndexFlatL2 quantizer by adding said centroids to it
        - create IndexIVFFlat with said quantizer + add points to it
    Note:
        - nlist == number of centroids / number of inverted lists
    '''
    
    nb, d = data_pts.shape # 

    total_start = time.time()
    # A) train K-means
    outer_km = faiss.Kmeans(d=d, k=nlist, niter=km_n_iter, verbose=verbose, max_points_per_centroid=km_max_pts)

    print(f"[1] Train K-means for {km_n_iter} iterations on {nb} points...")

    start_time_train = time.time()
    outer_km.train(data_pts)  # Train with specified number of iterations
    end_time_train = time.time()

    # B) K-means quantizer ( to search the centroids )
    print(f"[2] Initialize quantizer of {d} dimensions, with {nlist} centroids...")
    outer_km_quantizer = faiss.IndexFlatL2(d)
    outer_km_quantizer.verbose = verbose

    quant_start_time_add = time.time()
    assert outer_km_quantizer.is_trained == True
    outer_km_quantizer.add(outer_km.centroids)  # Add centroids directly to quantizer
    quant_end_time_add = time.time()

    # C) Create a new IVF index with the fresh quantizer
    print(f"[3] IndexIVFFLat: Adding {nb} points of {d} dimensions, to {nlist} centroids from the quantizer...")
    outer_index = faiss.IndexIVFFlat(outer_km_quantizer, d, nlist, faiss.METRIC_L2)
    outer_index.verbose = verbose

    outer_index.train(data_pts)
    start_time_add = time.time()
    outer_index.add(data_pts)    # Add data to index
    end_time_add = time.time()

    total_end = time.time()
    print(f"\n[-] Finished get_faiss_ivfflatL2_from_individual_components")
    print(f"- TRAIN: K-means took {round(end_time_train - start_time_train, 3)} s")
    print(f"- ADD: centroids to quantizer {round(quant_end_time_add - quant_start_time_add, 3)} s")
    print(f"- ADD: points to index took {round(end_time_add - start_time_add, 3)} s")
    print(f"- TOTAL time: {round(total_end - total_start, 3)} s")
    print(f"- Invlist Imbalance Factor: {outer_index.invlists.imbalance_factor()}")
    print()
    return outer_index


################################################# 
# Index Internals
################################################# 

# def get_centroid_ids_per_query(index, queries, nprobe=10, plot_hist=False):
#     _, cids = index.quantizer.search(queries, nprobe)
#     num_unique = len(np.unique(cids))
#     print(f"- Total number of unique centroids visited: {num_unique}")
#     print(f"- % of total centroids: {100*round(num_unique/index.nlist,2)} %")
#     if plot_hist:
#         unique_values, counts = np.unique(cids.flatten(), return_counts=True)
#         plt.bar(unique_values, counts, width=5, color='skyblue', edgecolor='black')
#         plt.xlabel('Centroid IDs')
#         plt.ylabel('Frequency')
#         plt.title('Histogram of Centroid Frequencies across queries')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.show()


def get_centroid_ids_per_query(index, queries, nprobe=10, plot_hist=False, sorted_order=False, dataset_name='unknown-dataset', save_plot=False, save_path="."):
    # Perform the quantization (search over the queries)
    _, cids = index.quantizer.search(queries, nprobe)
    nqueries = len(queries)
    n_centroids = index.nlist
    counts = np.zeros(n_centroids, dtype=int)

    # Update counts based on the cids
    visited_cids, cid_counts = np.unique(cids.flatten(), return_counts=True)
    counts[visited_cids] = cid_counts  # Set count for visited centroids

    num_unique = len(visited_cids)
    print(f"- Total number of unique centroids visited: {num_unique}")
    print(f"- % of total centroids: {100 * round(num_unique / n_centroids, 2)} %")
    
    if plot_hist:
        # Sort by frequency if sorted_order=True
        if sorted_order:
            sorted_indices = np.argsort(counts)[::-1]  # Sort by descending frequency
            sorted_counts = counts[sorted_indices]
        else:
            sorted_counts = counts

        # Plot the frequency
        plt.plot(
            sorted_counts, 
            marker='o', 
            linestyle='-', 
            linewidth=1, 
            alpha=0.8,
            # label=f"{dataset_name} (nprobe={nprobe}, nqueries={nqueries})"
            label=f"{dataset_name}"
        )
        
        plt.xlabel('Cluster_id (sorted by freq)')
        plt.ylabel('Frequency')
        plt.title(f'Cluster Frequency')
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend outside the plot
        plt.tight_layout()  # Adjust layout to prevent clipping

        
        if save_plot: 
            print(f"Storing skew plot in '{save_path}/{dataset_name}.png'")
            plt.savefig(f"{save_path}/{dataset_name}.png")
        else:
            plt.show(block=False)
    
    return visited_cids



#################################################################
# Training
#################################################################

def train_ivfflat(
    data,
    nlist=10,
    km_n_iter=10,
    km_max_pts=256, # faiss default
    seed=1,
    nredo=1,
    verbose=True,
    store_dir=None,
    metric='euclidean',
    store = False,
    train_pts=None,
):
    if verbose: print(f"Kmeans... {nlist=} {km_n_iter=} {km_max_pts=} {seed=} {nredo=} {metric=}")
    nb, d = data.shape

    if metric == "euclidean" or metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    elif metric == "angular" or metric == faiss.METRIC_INNER_PRODUCT or metric == "dot":
        #NOTE: make sure data has been normalized if angular (cosine), if dot/mips no need
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    index.cp.seed = seed
    index.cp.niter = km_n_iter
    index.cp.max_points_per_centroid = km_max_pts
    index.cp.nredo = nredo
    index.verbose = verbose

    strain = time.perf_counter()
    if train_pts is not None:
        index.train(train_pts)
    else:
        index.train(data)
    etrain = time.perf_counter()

    sadd = time.perf_counter()
    index.add(data)
    eadd = time.perf_counter()

    if store_dir is not None:
        filename = (
            store_dir
            + f"/index-n_iter_{km_n_iter}-nlist_{nlist}-max_pts_{km_max_pts}-seed_{seed}.index"
        )
        if store:
            print("storing index", filename)
            faiss.write_index(index, filename)
    
    if verbose: print(f"\t---> Index Train Time = {(etrain - strain)*1000} ms | Add Time = {(eadd - sadd)*1000} ms <---")
    return index, etrain - strain, eadd - sadd


################################################# 
# Creating Datasets & Dataset handling
################################################# 


# BOWEN : GET DATA <---
def load_dataset(dbname="SIFT1M"):
    print(f"Loading Dataset = {dbname}")
    if dbname.startswith("SIFT"):
        # big-ann sift1b, which is sliced 1m, 10m, 100m etc
        dbsize = int(dbname[4:-1])  # SIFT1M to SIFT1000M

        xb = mmap_bvecs(os.path.join(f"/pub/scratch/{username}/datasets/bigann", 'bigann_base.bvecs'))
        xq = mmap_bvecs(os.path.join(f"/pub/scratch/{username}/datasets/bigann", 'bigann_query.bvecs'))
        gt = ivecs_read(os.path.join(f"/pub/scratch/{username}/datasets/bigann", 'gnd/idx_%dM.ivecs' % dbsize))

        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        xq = xq.astype("float32").copy()
        xq = np.array(xq, dtype=np.float32)
        gt = np.array(gt, dtype=np.int32)
        metric = 'euclidean'
    
    elif dbname.startswith("DEEP"):
        xb, xq, gt, metric = load_deep10M(f"/pub/scratch/{username}/datasets/ann-fvecs/deep-96-angular")
    
    
    # BOWEN: ann benchmark : https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file
    elif dbname in [
        "deep-image-96-angular", 
        "glove-25-angular",
        "glove-50-angular",
        "glove-100-angular",
        "sift-128-euclidean", # != SIFTXM which is from big-ann-benchmarks, use the other one
        "lastfm-64-dot",
    ]:
        # Use ANNBENCHMARKS loader
        D, dimension = get_dataset_ann_benchmarks(dbname)
        xq, xb, gt, xt, (nq, d) = D['test'], D['train'], D['neighbors'], D['train'], D['test'].shape
        xq, xb, gt, xt, = np.array(xq), np.array(xb), np.array(gt), np.array(xt),
        metric = dbname.split('-')[-1]  # Extract metric from dataset name
    
    else:
        print("Unknown dataset:", dbname, file=sys.stderr)
        sys.exit(1)
    
    # 23/02/25 --- https://github.com/erikbern/ann-benchmarks/issues/574, deep is euclidean, i believe annbench has it wrongly as angular
    if dbname == "deep-image-96-angular":
        dbname = "deep-image-96-euclidean"
        metric = "euclidean" # just doing some tests...

    if metric == "angular":
        print(f"normalizing for {dbname} because metric = {metric}")
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
    return xb, xq, gt, metric


def get_dataset_fn_ann_benchmarks(dataset_name: str) -> str:
    # from ann-benchmarks
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    BASE_DATA_FOLDER=f'/pub/scratch/{username}/vdb-project-data'
    if not os.path.exists(f"{BASE_DATA_FOLDER}/data/datasets"):
        os.mkdir(f"{BASE_DATA_FOLDER}/data/datasets")
    return os.path.join(f"{BASE_DATA_FOLDER}/data/datasets", f"{dataset_name}.hdf5")

def download_ann_benchmarks(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if os.path.exists(destination_path):
        print(f"File already exists at {destination_path}. Skipping download.")
        return

    print(f"Downloading {source_url} -> {destination_path}...")
    try:
        with requests.get(source_url, stream=True) as response:
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
            total_size = int(response.headers.get('content-length', 0))
            with open(destination_path, 'wb') as file, tqdm(
                desc=destination_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks
                    file.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:
        print(f"Error downloading {source_url}: {e}")


def get_dataset_ann_benchmarks(dataset_name: str) -> Tuple[h5py.File, int]:
    # from ann_benchmarks
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn_ann_benchmarks(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download_ann_benchmarks(dataset_url, hdf5_filename)
    except:
        print(f"Cannot download {dataset_url}")

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension


def get_skewed_dataset(xb, skew_on_centroid=0, nlist=1000, compute_GT = True, nearest_cents_to_include=2, seed=None, plot_skew=False, plt_skew_nprobe=5, print_skew=True):
    '''
    # xb, xq, xt, gt, metric = load_sift1M(f"/pub/scratch/vmageirakos/datasets/ann-fvecs/sift-128-euclidean")
    # # if you want skewed dataset
    # selected_vectors_ids, selected_vectors, gt_selected_vectors, trained_index = get_skewed_dataset(xb, 
    #                                                                                             skew_on_centroid=0, 
    #                                                                                             nlist=1000, 
    #                                                                                             compute_GT = True, 
    #                                                                                             nearest_cents_to_include=10
    #                                                                                             )
    # # _ = analyze_cluster_distribution(trained_index)
    # selected_vectors.shape
    # vvids = get_centroid_ids_per_query(
    #     # index_ivf, 
    #     trained_index, 
    #     selected_vectors, 
    #     # selected_vectors[0:1,:], 
    #     nprobe=20, 
    #     # nprobe=15, 
    #     plot_hist=True,
    #     sorted_order=True
    #     )
    '''
    d = xb.shape[1]
    
    if seed is None:
        seed = int(time.time())

    ### init trained index
    # ndex for coarse quantization
    trained_quantizer = faiss.IndexFlatL2(d)  # flat index for coarse quantization
    trained_index = faiss.IndexIVFFlat(trained_quantizer, d, nlist, faiss.METRIC_L2)
    trained_index.cp.seed = seed
    trained_index.cp.niter= 25
    trained_index.cp.nredo = 1
    trained_index.verbose=True
    
    # Train the IVF index
    trained_index.train(xb)
    trained_index.add(xb)

    # Chose vectors from nearby clusters ( ~SKEWED DATASET HERE <<<<<< ? )
    centr_cluster = trained_index.quantizer.reconstruct_n()[skew_on_centroid]
    nearest_cents = trained_index.quantizer.assign(centr_cluster.reshape(1, -1), nearest_cents_to_include)
    nearest_cents.shape
    
    # ids, codes = get_invlist(trained_index.invlists, 10)
    selected_vectors_ids, codes = get_points_in_invlists(trained_index, nearest_cents.ravel())
    selected_vectors = codes.view("float32")
    #### compute ground truth
    if compute_GT:
        # selected_vectors.shape
        k = 100
        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        _, gt_selected_vectors = gt_index.search(selected_vectors, k)

    if plot_skew:
        _ = get_centroid_ids_per_query(
            index=trained_index, # D(x)
            # combined_index, # D(x), Q(x) 
            queries=selected_vectors,#[np.random.choice(xq.shape[0], size=5, replace=False), :],
            nprobe=plt_skew_nprobe, 
            plot_hist=True,
            sorted_order=True
        )
    if print_skew:
        _ = get_centroid_ids_per_query(
            index=trained_index, # D(x)
            # combined_index, # D(x), Q(x) 
            queries=selected_vectors,#[np.random.choice(xq.shape[0], size=5, replace=False), :],
            nprobe=plt_skew_nprobe, 
            plot_hist=False,
            sorted_order=True
        )

    return selected_vectors_ids, selected_vectors, gt_selected_vectors, trained_index


def read_ivecs(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def read_fvecs(fname):
    return read_ivecs(fname).view('float32')

def load_sift1M(PATH):
    """
    Load SIFT1M from fvecs data. 
    
    If in system group servers:
        DATA_PATH = f"/pub/scratch/vmageirakos/datasets/ann-fvecs/sift-128-euclidean"
    If local macbook:
        DATA_PATH = "/Users/mageirakos/Documents/projects/crack-vdb/data/" # local
    """
    print("Loading ann benchmarks sift1M...", end='', file=sys.stderr)
    xt = read_fvecs(f"{PATH}/sift_learn.fvecs")
    xb = read_fvecs(f"{PATH}/sift_base.fvecs")
    xq = read_fvecs(f"{PATH}/sift_query.fvecs")
    gt = read_ivecs(f"{PATH}/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)
    metric = 'euclidean'
    print("dataset shape:")
    print(f"{xb.shape=}")
    print(f"{gt.shape=}")
    print(f"{xq.shape=}")
    xb, xq, xt, gt = np.array(xb), np.array(xq), np.array(xt), np.array(gt)
    return xb, xq, xt, gt, metric


def load_deep10M(dataset_path):
    """Loads DEEP10M dataset files."""
    # dont remember which gt i downloads -->> use ann bench instear for paper but switch metric to be euclidean,
    # others:, 
    # - https://disk.yandex.ru/d/11eDCm7Dsn9GA
    # - https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py#L354
    # - https://github.com/matsui528/deep1b_gt
    # - https://big-ann-benchmarks.com/neurips21.html
    print("Loading deep10M...", end='', file=sys.stderr)
    xb = read_fvecs(f"{dataset_path}/deep10M.fvecs")
    xq = read_fvecs(f"{dataset_path}/deep1B_queries.fvecs")
    gt = read_ivecs(f"{dataset_path}/deep1B_groundtruth.ivecs")
    print("done", file=sys.stderr)
    metric = 'euclidean'
    return np.array(xb), np.array(xq), np.array(gt), metric

def create_symlink(target_dir, symlink_name):
    '''
    Creates a symlink named symlink_name that points to target_dir
    '''
    try:
        os.symlink(target_dir, symlink_name)
        print(f"Symlink created: {symlink_name} -> {target_dir}")
    except FileExistsError:
        print(f"Symlink {symlink_name} already exists.")
    except Exception as e:
        print(f"Error creating symlink: {e}")


def remove_symlink(symlink_name):
    '''
    Removes the symlink named symlink_name
    '''
    try:
        os.remove(symlink_name)  # Or os.unlink(symlink_name)
        print(f"Symlink {symlink_name} removed.")
    except FileNotFoundError:
        print(f"Symlink {symlink_name} does not exist.")
    except Exception as e:
        print(f"Error removing symlink: {e}")


def compute_and_save_ground_truth(dataset_vectors, query_vectors, k, ground_truth_file, use_gpu=False):
    """
    Computes the ground truth for a given query set by finding the nearest neighbors
    from a dataset using FAISS and saves the result to a file. Supports optional GPU acceleration.
    
    Parameters:
    - dataset_vectors: numpy array of shape (n_data, d), where n_data is the number of data points and d is the dimensionality.
    - query_vectors: numpy array of shape (n_queries, d), where n_queries is the number of query vectors.
    - k: number of nearest neighbors to retrieve.
    - ground_truth_file: path to the file where the ground truth (indices of nearest neighbors) will be saved.
    - use_gpu: boolean, if True, utilizes GPU for faster computation if available.
    """
    # import faiss
    
    # Step 1: Build the FAISS index
    d = dataset_vectors.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(d)  # L2 distance metric (Flat index)
    
    if use_gpu:
        # Move the index to GPU
        res = faiss.StandardGpuResources()  # Initialize GPU resources
        index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 refers to the first GPU device
    
    index.add(dataset_vectors)  # Add dataset vectors to the index

    # Step 2: Perform the search for each query
    distances, indices = index.search(query_vectors, k)  # distances and indices of nearest neighbors
    
    # Step 3: Save the ground truth (indices of nearest neighbors)
    print(f"Saving ground truth to: {ground_truth_file}")
    np.save(ground_truth_file, indices)  # Save the indices as a .npy file
    
    # Optionally, you can also save the distances if needed
    # np.save(ground_truth_file.replace('.npy', '_distances.npy'), distances)

    print(f"Ground truth (indices) saved to: {ground_truth_file}")
    return 


def ivecs_read(fname):
    """
    Used to read the ground truth file in ivecs format
    """
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Example format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def create_random_dataset_fast(d=16, nb=100, nq=1, seed=1234):
    '''
    This is faster than calling SyntheticDataset from FAISS which returns ground truth etc.
    '''
    # nq = 1                       # nb of queries
    np.random.seed(seed)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    print("dataset shape:")
    print(f"{xb.shape=}")
    print(f"{xq.shape=}")
    return xb, xq


def create_synthetic_dataset(d=16,nt=100,nb=100,nq=2,metric='L2'):
    # d=16
    # nt=100
    # nb=100
    # nq=2
    metric='L2'
    ds = SyntheticDataset(d=d,nt=nt,nb=nb,nq=nq,metric=metric)
    xb = ds.get_database()
    gt = ds.get_groundtruth()
    xq = ds.get_queries()
    xt = ds.get_train()
    print("dataset shape:")
    print(f"{xb.shape=}")
    print(f"{gt.shape=}")
    print(f"{xq.shape=}")
    return ds, xb, gt, xq, xt

def shuffle_queries(xq, gt, seed=42):
    # Set up random seed and train size
    random.seed(seed)
    np.random.seed(seed)

    # Generate a permutation of indices to shuffle both xq and gt identically
    total_samples = xq.shape[0]
    shuffled_indices = np.random.permutation(total_samples)

    # Apply the permutation to both xq and gt
    xq_shuffled = xq[shuffled_indices]
    gt_shuffled = gt[shuffled_indices]
    return xq_shuffled, gt_shuffled

def increase_queries_by_duplicates(xq, gt, factor=2):
    '''
    Takes as input a numpy array of queries (xq) and increases it by a factor.
    This is the number of duplicates of xq in the total, as it increases it by concating at the end
    '''
    if factor < 1:
        raise ValueError("factor must be at least 1")
    return np.tile(xq, (factor, 1)), np.tile(gt, (factor, 1))

def compute_and_save_ground_truth(dataset_vectors, query_vectors, k, ground_truth_file, metric="euclidean"):
    """
    Computes the ground truth for a given query set by finding the nearest neighbors
    from a dataset using FAISS and saves the result to a file.
    
    Parameters:
    - dataset_vectors: numpy array of shape (n_data, d), where n_data is the number of data points and d is the dimensionality.
    - query_vectors: numpy array of shape (n_queries, d), where n_queries is the number of query vectors.
    - k: number of nearest neighbors to retrieve.
    - ground_truth_file: path to the file where the ground truth (indices of nearest neighbors) will be saved.
    """
    
    # Step 1: Build the FAISS index
    d = dataset_vectors.shape[1]  # Dimension of the vectors
    if metric == "euclidean":
        index = faiss.IndexFlatL2(d)  # L2 distance metric
    elif metric =="angular":
        dataset_vectors = dataset_vectors / np.linalg.norm(dataset_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(d)  # L2 distance metric

    index.add(dataset_vectors)  # Add dataset vectors to the index

    # Step 2: Perform the search for each query
    distances, indices = index.search(query_vectors, k)  # distances and indices of nearest neighbors
    
    # Step 3: Save the ground truth (indices of nearest neighbors)
    print(f"{ground_truth_file=}")
    np.save(ground_truth_file, indices)  # Save the indices as a .npy file
    
    # Optionally, you can also save the distances if needed
    # np.save(ground_truth_file.replace('.npy', '_distances.npy'), distances)

    print(f"Ground truth (indices) saved to: {ground_truth_file}")
    return 

def generate_perturbed_vectors(input_vectors, epsilon, target_num_queries):
    """
    Generate perturbed vectors for randomly selected vectors from the input dataset until 
    the total number of vectors reaches target_num_queries.

    :param input_vectors: The input array of vectors (2D numpy array with shape (n, dimensions)).
    :param epsilon: The perturbation magnitude.
    :param target_num_queries: The target number of perturbed vectors to generate.
    :return: A numpy array of perturbed vectors with shape (target_num_queries, dimensions).
    """
    n_vectors, dimensions = input_vectors.shape

    if target_num_queries < n_vectors:
        raise ValueError("target_num_queries must be greater than or equal to the number of input vectors.")

    # Initialize an array to hold the perturbed vectors
    perturbed_vectors = []

    # Randomly select vectors and generate perturbations until the target is reached
    while len(perturbed_vectors) < target_num_queries:
        # Randomly sample vectors from the input dataset
        indices = np.random.choice(n_vectors, size=min(target_num_queries - len(perturbed_vectors), n_vectors), replace=False)
        sampled_vectors = input_vectors[indices]

        # Generate Gaussian perturbations for the sampled vectors
        perturbations = np.random.normal(loc=0, scale=epsilon, size=sampled_vectors.shape)
        perturbed_batch = sampled_vectors + perturbations

        # Append perturbed vectors to the list
        perturbed_vectors.extend(perturbed_batch)

    # Trim the array to the exact target_num_queries size
    perturbed_vectors = np.array(perturbed_vectors[:target_num_queries])

    return perturbed_vectors

################################################# 
# Plots & Visualization
################################################# 

def plot_hist_invlists_faiss_index(index):
    # invlist sizes histogram
    bc = np.bincount([index.invlists.list_size(l) for l in range(index.invlists.nlist)])
    plt.step(np.arange(bc.size), bc)
    plt.xlabel('size of invlist')
    plt.ylabel('nb of invlists')
    plt.grid()
    plt.show()


def analyze_cluster_distribution(index, NAME="", plt_sorted=False):
    """
    Efficiently analyzes the distribution of points across centroids in a FAISS index
    using direct SWIG access to internal assignments.
    
    Args:
        index: Trained FAISS index (IVF-based)
        NAME: String to prefix plot titles
    Returns:
        dict: Mapping of centroid_id to number of points
    """
    # Get the inverted lists object
    invlists = faiss.extract_index_ivf(index).invlists
    
    # Initialize counts dictionary
    centroid_distribution = {}
    
    # Iterate through all lists to get counts
    for i in range(invlists.nlist):
        centroid_distribution[i] = invlists.list_size(i)
    
    # Create figure with four subplots
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Histogram of counts
    counts = list(centroid_distribution.values())
    ax1.hist(counts, bins=50, edgecolor='black')
    ax1.set_title(f'{NAME}\n - Distribution of Points per Centroid')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Frequency')
    
    # CDF of counts
    counts_sorted = sorted(counts)
    cumulative = np.arange(1, len(counts_sorted) + 1) / len(counts_sorted)
    ax2.plot(counts_sorted, cumulative, 'b-')
    ax2.set_title(f'{NAME}\n - CDF of Points per Centroid')
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('Cumulative Probability')
    ax2.grid(True)
    
    if plt_sorted:
        # Histogram by cluster ID (sorted by cluster size)
        sorted_clusters = sorted(centroid_distribution.items(), key=lambda x: x[1])  # Sort by number of points
        cluster_ids, cluster_sizes = zip(*sorted_clusters)  # Unpack sorted items

        # Generate a new x-axis based on the sorted order
        sorted_x_axis = range(len(cluster_sizes))  # Sequential indices for sorted data

        # Plot with sorted x-axis
        ax3.bar(sorted_x_axis, cluster_sizes, width=1.0, edgecolor='black')
        ax3.set_title(f'{NAME}\n - Points per Cluster ID (Sorted by Cluster Size)')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Points')
    else:
        # Histogram by cluster ID
        cluster_ids = list(centroid_distribution.keys())
        cluster_sizes = list(centroid_distribution.values())
        ax3.bar(cluster_ids, cluster_sizes, width=1.0, edgecolor='black')
        ax3.set_title(f'{NAME}\n - Points per Cluster ID')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Points')

    
    # CDF by cluster ID
    sorted_by_id = sorted(centroid_distribution.items())
    cluster_ids_sorted = [x[0] for x in sorted_by_id]
    counts_by_id = [x[1] for x in sorted_by_id]
    cumulative_by_id = np.cumsum(counts_by_id) / sum(counts_by_id)
    ax4.plot(cluster_ids_sorted, cumulative_by_id, 'r-')
    ax4.set_title(f'{NAME}\n - CDF by Cluster ID')
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Cumulative Proportion')
    ax4.grid(True)
    
    # Add some statistics as text
    stats_text = f"""
    Total Points: {sum(counts):,}
    Mean points/centroid: {np.mean(counts):.1f}
    Median points/centroid: {np.median(counts):.1f}
    Std Dev: {np.std(counts):.1f}
    Min: {min(counts)}
    Max: {max(counts)}
    Empty centroids: {sum(1 for x in counts if x == 0)}
    """
    fig.text(1- 0.02, 1- 0.02, stats_text, fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return centroid_distribution

################################################# 
# Evaluation & Metrics
################################################# 

def compute_recall(IDX, GRND_TRTH, k):
    """
    Compute the Recall@k from results of FAISS index
    input: 2 sets of pids to compare
    shapes shoudl be (N, dim) N: num points, dim: dimensions fo vectors
    """
    recall = 0
    for i in range(IDX.shape[0]):
        # Compare top-k indices with ground truth
        relevant = GRND_TRTH[i][:k]  # Get relevant items from ground truth
        retrieved = IDX[i][:k]  # Get top-k retrieved items
        # Check how many relevant items are in the retrieved items
        recall += len(set(retrieved) & set(relevant))
    
    return recall / (IDX.shape[0] * k)

def calculate_recall(I, gt, k):
    """
    I: ANN search result. numpy array of shape (nq, ?)
    gt: numpy array of shape (>=nq, ?)
    """
    assert I.shape[1] >= k
    assert gt.shape[1] >= k
    nq = I.shape[0]
    total_intersect = 0
    for i in range(nq):
        n_intersect = np.intersect1d(I[i, :k], gt[i, :k], assume_unique=False, return_indices=False).shape[0]
        total_intersect += n_intersect
    return total_intersect / (nq * k)

def print_recall(I, gt): 
    """
    print recall (depends on the shape of I, return 1/10/100)
    """
    k_max = I.shape[1]
    if k_max >= 100:
        k_set = [1, 10, 100]
        print(' ' * 4, '\t', 'R@1    R@10   R@100')
    elif k_max >= 10:
        k_set = [1, 10]
        print(' ' * 4, '\t', 'R@1    R@10')
    else:
        k_set = [1]
        print(' ' * 4, '\t', 'R@1')
    for k in k_set:
        print("\t{:.4f}".format(compute_recall(I, gt, k)), end=' ')
    print()





################################################# 
# OTHER AND/OR NOT WORKING ( but maybe can fix )
################################################# 

'''
Wanted to have a prototype .crack/.refine/.reorg in python
--> used vstack to combine invlists
PROB 1) --> vstack is read only won't allow reassignments
PROB 2) --> faiss.clone_index(combined_index) does not work with vstack
PROB 2) --> return combined_index is not correct, the internals afterwards are wrong
    - if you do combined_index.invlists.nlist in the function != after the return <<< 

Probably best to just go do a C++ implementation and skip the python prototype
'''

def reorg():
    return

def refine():
    return

def crack_single(index_to_crack, crack_query, assignments=None, metric='euclidean'):
    '''
    NOTE: Relies on custom faiss binding/build from source
    - needs .add_empty_list() implemented for ArrayInvertedLists

    TODO:
        - handle assignments
        - don't do get c_visited [reconstruct_n()] twice (also done outside crack_single) --> also needed by refine if you refine { combine effort }
        - don't do replace_ivf_quantizer twice ( also done outside crack signle) ---> should be part of re-org

    params:
    - index_to_crack: the index we're cracking, need to return a new cracked index
        - points expected to be in index_to_crack
    - crack_query: crack location, basically the new centroid
    '''
    # assert index_to_crack.own_invlists == False # otherwise segfault?

    # create a new crack
    # if metric == "euclidean":
    #     new_crack = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 1, faiss.METRIC_L2)
    # elif metric == "angular":
    #     new_crack = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1, faiss.METRIC_INNER_PRODUCT)
    # new_crack.cp.min_points_per_centroid = 1 # to avoid warning
    # new_crack.train(crack_query) # add the new crack

    # assert new_crack.nlist == 1

    # add empty invlist 
    faiss.downcast_InvertedLists(index_to_crack.invlists).add_empty_list()
    index_to_crack.nlist = index_to_crack.invlists.nlist # change .nlist 

    # assert index_to_crack.nlist == index_to_crack.invlists.nlist
    # assert index_to_crack.ntotal == index_to_crack.invlists.compute_ntotal(), f"{index_to_crack.ntotal=} != {index_to_crack.invlists.compute_ntotal()=}"
    
    # change quantizer (combine old centroids w/ new crack where centroid == query)  
    # - combine centroids
    # crack query has appended at the end, and has id = len(prev_centroids) + 1
    # replace quantizer TODO: turn it into function, it's also called again by others
    
    if metric == "euclidean":
        new_quantizer = faiss.IndexFlatL2(index_to_crack.d)
    elif metric == "angular":
        new_quantizer = faiss.IndexFlatIP(index_to_crack.d)
    # combined_centroids = np.concatenate([index_to_crack.quantizer.reconstruct_n(), crack_query])
    combined_centroids = np.concatenate([index_to_crack.quantizer.reconstruct_n(), crack_query], axis=0)
    new_quantizer.add(combined_centroids)

    # assert new_quantizer.ntotal == index_to_crack.nlist, f"{new_quantizer.ntotal=} != {index_to_crack.nlist=}"
    # NOTE: this takes a while & we do it again in .refine() is there a way to avoid doing it twice?
    _ = replace_ivf_quantizer(index_to_crack, new_quantizer)


    # print(f"- After crack, {index_to_crack.nlist=}")

    return index_to_crack


# TODO: Test
# BUG: cracking does not work with VSTack I need to modify ArrayInvertedLists implementation in the FAISS C++ code to do what I wish
def not_crack_single(index_to_crack, crack_loc, assignments=None, metric='euclidean'):
    '''
    Add a single new partition at index_to_crack, defined by crack_loc. 
    If assignments=None, then this new partition will be empty.
    Use assignments = None, if you plan to .refine() afterwards which will handle new assignments...
    
    params:
    - index_to_crack: the index we're cracking, need to return a new cracked index
        - points expected to be in index_to_crack
    - crack_loc: crack location, basically the new centroid
    '''
    print("THIS IS WRONG YOU NEED TO MOFIDY FAISS CODEBASE")
    exit()
    # assert index_to_crack.own_invlists == False # otherwise segfault?

    # create a new crack
    d = index_to_crack.d
    if metric == "euclidean":
        new_crack = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 1, faiss.METRIC_L2)
    elif metric == "angular":
        new_crack = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1, faiss.METRIC_INNER_PRODUCT)
    new_crack.train(crack_loc) # add the new crack

    assert new_crack.nlist == 1
    #     # TODO: make assignments work. Q: should .remove() from assignments for index to crack happen here or before crack function?
    #       as assignments are pts that belonged to other invlists in index to crack and now go to crack loc invlists
    # if assignments != None:
    #     new_crack.add(assignments)
    
    # combine invlists:
    ilv = faiss.InvertedListsPtrVector() # a pointer for multiple invlists
    for index in [index_to_crack, new_crack]:
        ilv.push_back(index.invlists)
    combined_invlists = faiss.VStackInvertedLists(ilv.size(), ilv.data())
    # print(f"{combined_invlists=}")

    assert combined_invlists.nlist == (new_crack.nlist + index_to_crack.nlist)
    ntotal_new = combined_invlists.compute_ntotal()
    assert ntotal_new == index_to_crack.ntotal
    
    # combine centroids
    combined_centroids = np.vstack([index_to_crack.quantizer.reconstruct_n(), crack_loc])
    
    # combine new empty index
    combined_nlist = combined_invlists.nlist
    
    print(f"--- {combined_invlists=} ---")
    combined_invlists = faiss.downcast_InvertedLists(combined_invlists)
    print(f"--- {combined_invlists=} ---")

    print()
    if metric == "euclidean":
        combined_quantizer = faiss.IndexFlatL2(d)
        combined_quantizer.add(combined_centroids)
        combined_index = faiss.IndexIVFFlat(combined_quantizer, d, combined_nlist, faiss.METRIC_L2)
    elif metric == "angular":
        combined_quantizer = faiss.IndexFlatIP(d)
        combined_quantizer.add(combined_centroids)
        combined_index = faiss.IndexIVFFlat(combined_quantizer, d, combined_nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"1) {combined_index.invlists.nlist=}")
    assert combined_index.is_trained == True

    # NOTE: true or fals for .own_invlists() ?
    combined_index.replace_invlists(combined_invlists, False) # BUG: not correct
    print(f"2) {combined_index.invlists.nlist=}")
    assert combined_index.invlists.nlist == combined_nlist
    print(f"{combined_index.invlists.nlist=}")
    # update .ntotal() not done automatically
    combined_index.ntotal = combined_invlists.compute_ntotal() 
    print(f"3) {combined_index.invlists.nlist=}")
    
    # assert index_to_crack.ntotal == combined_index.ntotal

    # delete the individual indexes { new_crack and index_to_crack } and return the combined index 
    del index_to_crack
    del new_crack
    index_to_crack = combined_index
    print(f"- After crack, {index_to_crack.nlist=}")
    print(f"4) {combined_index.invlists.nlist=}")
    print(f"4) {index_to_crack.invlists.nlist=}")
    print(f"4) {index_to_crack=}")

    return faiss.clone_index(combined_index)


def search_crack_refine_single(outer_index, query, nprobe=10, k=10, refine_niter=1, refine_nprobe=1, CRACK=True, REFINE=True, DEBUG=False, metric="euclidean"):
    '''
    IF CRACK:
        - create new centroid
        - & reorg

    IF REFINE:
        - refine local centroids 
        - & reog
    
    - refine_nprobe : nprobe closest centroids to .refine() (local kmeans for)
    '''
    # BUG CHEF NOTE: I thinkthere's a memory leak in add_preassigned
    timings = {
        "search": [],
        "get_c_visited": [],
        "get_p_visited": [],
        "inner_km_train": [],
        "inner_point_assignments": [],
        "replace_quantizer": [],
        "update_invlists": [],
        "crack": [],
    }
    
    d = outer_index.quantizer.d
    refine_nprobe = max(refine_nprobe, nprobe)
    # print(f"{refine_nprobe=}")
    # print(f"{outer_index.nlist=}, {refine_nprobe_factor=}, {nprobe=}")

    ############### SEARCH ###############
    # 1. search
    # query = query.reshape(1, -1)
    # search with original nprobe, and train with factor*orig_nprobe
    outer_index.nprobe = nprobe
    # print(f"{outer_index.nprobe=}")
    start_time = time.perf_counter()  
    D_p, I_p = outer_index.search(query, k) #TODO: see if I can get c visited, get p visited and assignemtns for q-means from .search() 
    end_time = time.perf_counter()
    timings["search"].append(end_time - start_time)
    

    # HACK to work around parallelism issue (the rest of the threads will have to wait )
    # NOTE: since we might have batch size, just pick the first query from the batch and crack/refine etc.
    # Alterantive handle cracking the entire thing (all queries in the batch)
    if query.shape[0] > 1:
        query = query[0:1,:]

    if CRACK:    
        ############### CRACK ###############
        # crack : create single new centroid at the query location. No assignments for now
        # NOTE: if no assignments, you should .refine() in order to fix the assignments...

        #TODO: assert crack_single works correctly, you get new centroid with id + 1 & previous ones are un-affected etc.
        
        outer_index.own_invlists = False # otherwise segfault
        # if query.shape 
        t0_crack = time.perf_counter()
        outer_index = crack_single(index_to_crack=outer_index, crack_query=query, assignments=None, metric=metric)
        t1_crack = time.perf_counter()
        timings["crack"].append(t1_crack - t0_crack)
    
    ############### REFINE ###############
    if not REFINE:
        return outer_index, D_p, I_p, timings, refine_nprobe
    
    # Query Loop
    assert refine_nprobe >= nprobe >= 1, f"MUST: {refine_nprobe=} >= {nprobe} >= {1}"
    outer_centroids = outer_index.quantizer.reconstruct_n() # or kmeans.centroids idk which is faster
    start_time = time.perf_counter()
    if CRACK:
        # if you cracked, you have an new empty partition which was inserted, that equals the query. So it'll be returns
        # thus +1 the refine_nprobe so c_visited_ids is unaffected
        # NOTE: the new crack will be in the area of the query, obviously, so you want to have it in the .refine()
        refine_nprobe += 1 
    c_visited_ids = outer_index.quantizer.search(query, refine_nprobe)[1][0]
    c_visited = outer_centroids[c_visited_ids]
    # print(f"{c_visited_ids=}, {outer_centroids.shape=}") # CHEF PRINT HERE << 
    end_time = time.perf_counter()
    timings["get_c_visited"].append(end_time - start_time)

    # 2a. get p_visited
    # NOTE: this is expensive operation currently O(pts_in_c_visited_ids) { the fewer the partition the larger this would be }
    start_time = time.perf_counter()
    pids_visited, p_visited = get_points_in_invlists(index=outer_index, invlists=c_visited_ids, DEBUG=DEBUG)
    # print(f"{pids_visited.shape=}, {p_visited.shape=}") # CHEF PRINT HERE << 
    # print(f"{outer_index.ntotal=}, {outer_index.nlist=}, {outer_index.invlists.nlist=}")
    end_time = time.perf_counter()
    timings["get_p_visited"].append(end_time - start_time)
    
    # 2c. initialize new inner_quantizer w/ c_visited centroids
    start_time = time.perf_counter()
    assert len(c_visited_ids) == refine_nprobe
    inner_km = faiss.Kmeans(d, refine_nprobe, niter=refine_niter, max_points_per_centroid=10_000) #NOTE:256 change max_points_per_centroids to know if you should subsample or not don't subsample inner km ( get inner pt assignments for free )
    inner_km.verbose = True
    # print(f"{c_visited.shape}")
    inner_km.train(p_visited, init_centroids=c_visited)
    end_time = time.perf_counter()
    timings["inner_km_train"].append(end_time - start_time)

    # inner_point_assignments
    start_time = time.perf_counter()
    inner_point_assignments = inner_km.index.assign(p_visited, k=1).reshape(1, -1)
    inner_point_assignments = c_visited_ids[inner_point_assignments]
    # prev_assignments = outer_index.quantizer.assign(p_visited, k=1).reshape(1, -1)
    end_time = time.perf_counter()
    timings["inner_point_assignments"].append(end_time - start_time)

    ############### REORG ###############
    # 2d. Replace quantizer
    start_time = time.perf_counter()
    outer_centroids[c_visited_ids] = inner_km.centroids
    if metric == "euclidean":
        new_quantizer = faiss.IndexFlatL2(d)
    elif metric == "angular":
        new_quantizer = faiss.IndexFlatIP(d)
    new_quantizer.add(outer_centroids)
    _ = replace_ivf_quantizer(outer_index, new_quantizer)
    end_time = time.perf_counter()
    timings["replace_quantizer"].append(end_time - start_time)

    start_time = time.perf_counter()
    outer_index.verbose = False # don't print
    # outer_index.remove_ids(pids_visited) # too slow
    for list_no in c_visited_ids:
        # BUG: this clears the invlist, but it does not change outer_index.ntotal() correctly
        # outer_index.ntotal -= outer_index.invlists.list_size(int(list_no)) # need to manualy handle this because add preassigned increases it but .resize() does not decrease it
        outer_index.invlists.resize(int(list_no), 0) 
    
    add_preassigned(index_ivf=outer_index, x=p_visited, a=inner_point_assignments.ravel(), ids=pids_visited.ravel()) 
    outer_index.ntotal -= len(p_visited) # add preassigned doesn't handle ntotal need to do it manually
    end_time = time.perf_counter()
    outer_index.verbose = True

    timings["update_invlists"].append(end_time - start_time)
    return outer_index, D_p, I_p, timings, refine_nprobe



#TODO: make cracking have max_iter per centroid locally to limit cracking
def run_crack_ivf(outer_index, queries, gt_ids, nprobe=10, k=10, inner_km_niter=1, limit_query=None, limit_crack=None, CRACK=False, dynamic_nprobe=False, plot_boxplot=False, DEBUG=False):
    """
    V0: Naive Python Crack IVF implementation.

    This is a naive first implementation of cracking using FAISS provided functions from swigg python bindings.
    CHEF NOTES:
        - NOTE: nprobe > 1 otherwise you can't crack...
        - BUG: Memory leak in add_preassigned()
        - NOTE: Low performance
        - DONE: Assert correctness of the algorithm (maybe a ug slipped but looks correct to me, other than the memory leak)
    
    ######################################################
    Algorithm Steps should be:
    For all Queries:
        1) .search() for the query
            - a) get results
            - b) get recall
            - c) get nprobe closest centroids -> "c_visited"
        IF any(c_visited.iter < max_local_iter ) ? CRACK : NO_CRACK
            - a) If NO_CRACK: return results
        2) if CRACK:
            - a) get all points in c_visited -> "p_visited"
            - b) use IDMap to keep track of -> point_ids & centroid_ids (outer<->inner)
            - c) initialize new inner_quantizer w/ c_visited centroids
                - single iteration of .train() on c_visited with p_visited
                - add(p_visited) using innuer_quantizer and get inner_invlists
            - d) update outer_index for the next query, to use the new:
                - inner_quantizer
                - inner_invlists 
    
    Notes on the algo step:
    - 1c) there's not cheap way to get it in python, there is a smart pointer that poitns to object which holds c_visited (called idx) when sub_search_func() is called in indexivf.cpp search
        -  I would have to change internals to optionally return this to have it "for free"
    """

    # Initialize a dictionary to store times for each section
    timings = {
        "search": [],
        "get_c_visited": [],
        "get_p_visited": [],
        "inner_km_train": [],
        "inner_point_assignments": [],
        "replace_quantizer": [],
        "update_invlists": [],
    }
    recalls, prcnt_pts_scanned = [], []
    orig_nprobe = nprobe
    orig_ntotal = outer_index.ntotal

    d = outer_index.quantizer.d
    if DEBUG: 
        print("[0] in run_crack_ivf:")
        print(f"- Total centroids : {outer_index.nlist}")
        print(f"- {queries.shape=}")

    # Query Loop
    outer_centroids = None
    if limit_query:
        assert limit_query <= len(queries), f"limit set too high, max is {len(queries)}"
        queries = queries[:limit_query].reshape(limit_query, -1)

    for qid, q in enumerate(queries):
        
        if limit_crack is not None and limit_crack <= qid:
            CRACK = False
        if outer_centroids is not None:
            assert np.array_equal(outer_centroids, outer_index.quantizer.reconstruct_n()), "After you set inner quantizer to outer quantizer this should have been equal"
            
        outer_centroids = outer_index.quantizer.reconstruct_n() # or kmeans.centroids idk which is faster
        
        if DEBUG: 
            print(f"Query ID = {qid}")
            print("[1] get NN to the query w/ .search()")
        
        # 1. search
        start_time = time.perf_counter()
        q = q.reshape(1, -1)
        # NOTE: Dynamic nprobe
        if dynamic_nprobe:
            if qid < 20:
                nprobe = int(min(outer_index.nlist, 2*orig_nprobe))
            else:
                nprobe = orig_nprobe
        
        outer_index.nprobe = orig_nprobe  
        D_p, I_p = outer_index.search(q, k)
        end_time = time.perf_counter()
        timings["search"].append(end_time - start_time)

        # 1b. get recall
        if DEBUG: 
            print("[1b] get Recall:")
            print(f"{I_p=}")
            print(f"{gt_ids[qid,:k].reshape(1,-1)=}")

        q_recall = compute_recall(I_p, gt_ids[qid,:].reshape(1,-1), k)
        recalls.append(q_recall)

        if DEBUG: print(f"- {qid=}) Recall@{k} = {q_recall}")

        # 1c. get c_visited
        if DEBUG: print("[1c] get centroids visited")
        start_time = time.perf_counter()
        c_visited_ids = outer_index.quantizer.search(q, nprobe)[1][0]
        c_visited = outer_centroids[c_visited_ids]
        end_time = time.perf_counter()
        timings["get_c_visited"].append(end_time - start_time)
        
        if DEBUG:
            print(f"- centroids visited for q{qid}: {c_visited_ids=}")
            # print(f"{c_visited=}")
            print("Cracking...")
    
        if not CRACK:
            continue
        
        ##################################### CRACK:
        
        # 2a. get p_visited
        if DEBUG: print("[2a-2b] get points visited & keep track of ids")
        start_time = time.perf_counter()
        pids_visited, p_visited = get_points_in_invlists(index=outer_index, invlists=c_visited_ids, DEBUG=DEBUG)
        end_time = time.perf_counter()
        timings["get_p_visited"].append(end_time - start_time)
        
        prcnt_pts_scanned.append(round(len(pids_visited)/outer_index.ntotal, 3))
        if DEBUG:
            print(f"- TOTAL points : {outer_index.ntotal}")
            print(f"- TOTAL points scanned (in c_visited) : {len(pids_visited)}")
            print(f"- % points scanned (in c_visited) : {round(len(pids_visited)/outer_index.ntotal,3)}")
            print(f"- All points visited for q{qid}: {pids_visited=}")
            print(f"- All assignment of points: {pids_visited=}")
            prev_assignments = outer_index.quantizer.assign(p_visited, k=1).reshape(1, -1) # expensive don't keep calling it
            print(f"- {prev_assignments=}")

        # 2c. initialize new inner_quantizer w/ c_visited centroids

        # if not CRACK:
        #     continue

        if DEBUG: print("[2c] train inned K-means from points visited")
        start_time = time.perf_counter()
        assert len(c_visited_ids) == nprobe
        inner_km = faiss.Kmeans(d, nprobe, niter=inner_km_niter) 
        inner_km.verbose = True
        inner_km.train(p_visited, init_centroids=c_visited)
        end_time = time.perf_counter()
        timings["inner_km_train"].append(end_time - start_time)

        # inner_point_assignments
        start_time = time.perf_counter()
        inner_point_assignments = inner_km.index.assign(p_visited, k=1).reshape(1, -1)
        inner_point_assignments = c_visited_ids[inner_point_assignments]
        # prev_assignments = outer_index.quantizer.assign(p_visited, k=1).reshape(1, -1)
        end_time = time.perf_counter()
        timings["inner_point_assignments"].append(end_time - start_time)
        if DEBUG: 
            print("[2c] initialize inner quantizer")
            print(f"- PREV (FULL) outer centroids : {outer_centroids=}")
            print(f"- PREV centroids visited: {outer_centroids[c_visited_ids]=}")
            print(f"- NEW centroids visited: {inner_km.centroids=}")
            print(f"- NEW point assignments (cid per pid) : {inner_point_assignments=}")

        # Skip replacement if no new assignment
        # NOTE: takes too long to get prev_assignments
        # TODO: figure out another way to early stop
        # if np.array_equal(inner_point_assignments, prev_assignments):
        #     continue

        # 2d. Replace quantizer
        if DEBUG: print("[2d] replace outer quantizer")
        start_time = time.perf_counter()
        outer_centroids[c_visited_ids] = inner_km.centroids
        new_quantizer = faiss.IndexFlatL2(d)
        new_quantizer.add(outer_centroids)
        _ = replace_ivf_quantizer(outer_index, new_quantizer)
        end_time = time.perf_counter()
        timings["replace_quantizer"].append(end_time - start_time)
        if DEBUG:
            print(f"- NEW (FULL) outer centroids : {outer_centroids=}")
            print(f"- NEW centroids visited (replaced outer) : {outer_centroids[c_visited_ids]=}")
            

# CHEF NOTE: no memory leak up until this point
# BUG: MEMORY LEAK SOMEWHERE IN ADD PREASSIGNED FUNCTION
        # Clear and update invlists
        if DEBUG: print("[2d] replace outer inverted lists")
        start_time = time.perf_counter()
        #### One way to clear the inv list:
        # for list_no in c_visited_ids:
        #     print("HEEEERE", list_no)
        #     # BUG: this clears the invlist, but it does not change outer_index.ntotal() correctly
        #     outer_index.invlists.resize(int(list_no), 0) 
        #### Another way with remove ids:
        
        # print(f"BEFORE REMOVE {outer_index.ntotal=}")
        # CHEF NOTE: If you remove this way make sure you keep track of ids correctly
        outer_index.remove_ids(pids_visited)
        # print(f"AFTER REMOVE {outer_index.ntotal=}")
        # TODO: Check that ids are correct when adding
        # BUG: MEMORY LEAK SOMEWHERE IN ADD PREASSIGNED FUNCTION ( MEMORY JUST KEEPS INCREASING )
        #   BUG NOTES:
        #       - 1) if you have omp.set_num_thread(1) it th leak is much slower
        # add_preassigned(outer_index, p_visited, inner_point_assignments.ravel())#, ids=pids_visited.ravel())
        if DEBUG:
            print("[BUG] There is a memory leak in add_preassigned() that should be fixed...")
            after_remove_pids, _ = get_points_in_invlists(index=outer_index, invlists=c_visited_ids, DEBUG=DEBUG)
            assert len(after_remove_pids) == 0, "Number of poitns after remove_ids in c_visited_ids should be 0"
            assert outer_index.ntotal == orig_ntotal - len(pids_visited), "ntotal at this stage should be #total_before - #removed pids"
            # print(f"point ids in c_visited {} after remove_ids: {}")
        
        # BUG: memory leak in add_preassigned somewhere
        add_preassigned(index_ivf=outer_index, x=p_visited, a=inner_point_assignments.ravel(), ids=pids_visited.ravel())
        end_time = time.perf_counter()

        if DEBUG:
            after_pids, _ = get_points_in_invlists(index=outer_index, invlists=c_visited_ids, DEBUG=DEBUG)
            # print(f"AFTER add_preassigned(): point ids in {c_visited_ids} : {after_pids}")
            assert set(after_pids) == set(pids_visited), "Point IDs after add_preassigned() should be the same as before (& in the same centroids ids)"
            assert outer_index.ntotal == orig_ntotal, "ntotal at this stage should be the same as at the start"
            #TODO: assert that pid "X" AFTER has the same codes as pid "X" BEFORE  (but I think it is fine because I pass p_visited and pids_visited as is in add preassigned)

        timings["update_invlists"].append(end_time - start_time)

    # Calculate and display median statistics
    print("\n[-] Finished run_crack_ivf:")
    print("--- Median Timings for Each Section ---")
    for section, times in timings.items():
        if times:
            median_time = np.median(times)
            print(f"- {section}: {median_time:.6f} seconds")
        else:
            print(f"- {section}: 0.0 seconds")
    
    print("\n--- Total Timings for Each Section ---")
    for section, times in timings.items():
        if times:
            total_time = np.sum(times)
            print(f"- {section}: {total_time:.6f} seconds")
        else:
            print(f"- {section}: 0.0 seconds")
    
    print(f"- Invlist Imbalance Factor : {outer_index.invlists.imbalance_factor()}")
    recalls = np.array(recalls)
    # print(f"{recalls=}")
    print(f"- Recalls@{k}: {np.median(recalls)=} - {np.mean(recalls)=} - {round(np.std(recalls),3)=}")
    prcnt_pts_scanned = np.array(prcnt_pts_scanned)
    # NOTE: I only get points scanned if I know pids ( which is only when I crack in this implementation )
    print(f"-% points scanned : {round(np.median(prcnt_pts_scanned),3)} (median) - {round(np.mean(prcnt_pts_scanned),3)} (mean) ")
    

    # Optional: Plot boxplot of timings
    if plot_boxplot:
        plt.figure(figsize=(10, 6))
        plt.boxplot(timings.values(), labels=timings.keys(), vert=False)
        plt.title("Timing Distribution for Each Section")
        plt.xlabel("Time (seconds)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
