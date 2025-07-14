from __future__ import print_function
import time
import faiss
import os
from datetime import datetime
import pandas as pd
import argparse
from collections import defaultdict
# ours
from CrackIVF import *
from vasili_helpers import *


# Function to handle baseline CSV logic
import os
import pandas as pd

def handle_baseline_csv(file_path, columns, clear_flag):
    # Ensure file_path is a file and not a directory (extra safety check)
    print(f"{clear_flag= }")
    if os.path.exists(file_path) and os.path.isfile(file_path):
        if clear_flag:
            try:
                os.remove(file_path)
                print(f"Existing CSV file '{file_path}' deleted. Creating new CSV.")
            except OSError as e:
                print(f"Error deleting file '{file_path}': {e}")

        # Create new CSV with specified columns
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
    elif os.path.exists(file_path) and not os.path.isfile(file_path):
        raise ValueError(f"'{file_path}' exists but is not a file. Aborting operation.")
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
        print(f"New CSV created at '{file_path}'")

    return df

def add_rows_to_csv(data_to_add, df, file_path, is_single_row=False, store=False):
    """
    Add rows to a DataFrame and optionally append them to a CSV file.

    Args:
        data_to_add: Single row (dict or list) or multiple rows (list of dicts/lists).
        df: The existing DataFrame.
        file_path: Path to the CSV file.
        is_single_row: True if `data_to_add` is a single row, False for multiple rows.
        store: Whether to append the new rows to the CSV file.
    
    Returns:
        Updated DataFrame.
    """
    # Convert input to DataFrame
    new_data_df = pd.DataFrame([data_to_add]) if is_single_row else pd.DataFrame(data_to_add)
    missing_in_df = set(new_data_df.columns) - set(df.columns)
    extra_in_df = set(df.columns) - set(new_data_df.columns)

    assert not missing_in_df and not extra_in_df, (
        f"Column mismatch:\n"
        f"Missing in existing DataFrame: {list(missing_in_df)}\n"
        f"Extra in new data: {list(extra_in_df)}"
    )

    # Remove columns with all NaN values and append new rows
    df = df.dropna(axis=1, how="all")
    df = pd.concat([df, new_data_df], ignore_index=True)
    
    # Optionally store in the CSV file
    if store:
        print(f"Adding rows to {file_path}")
        new_data_df.to_csv(
            file_path, mode="a", index=False, header=not os.path.exists(file_path)
        )
    
    return df


# Example usage


parser = argparse.ArgumentParser()

# read & store results & dataset parameters
parser.add_argument(
    "--result_path",
    type=str,
    default="./results/",
    help="path to store results + indexes ",
)
parser.add_argument(
    "--baselines_path",
    type=str,
    default="./results/SIFT1M_default_BASELINES/baseline_results.csv",
    help="The path where the baseline results are stored",
)
parser.add_argument(
    "--dbname",
    type=str,
    default="SIFT1M",
    help="dataset name, e.g., SIFT100M, (detulat)",
)
parser.add_argument(
    "--skew",
    type=str,
    default="default",
    help="Skewed vs default/uniform skew (skew = default means use the XQ provided with the dataset. ['max', 'high', 'medium'])",
)
parser.add_argument(
    "--store",
    action="store_true",
    help="Enable this to store index and dataframe",
)
parser.add_argument(
    "--target_queries",
    type=int,
    default=0,
    help="Target number of queries ( 0 which means, it will just run whatever the dataset provides)",
)
# TODO; plot the skew << this has to be done during the run where you have access to the index
parser.add_argument(
    "--plot",
    action="store_true",
    default=False,
    help="If flag passes, then create and store plots (default False)",
)
parser.add_argument(
    "--get_qps",
    action="store_true",
    default=False,
    help="If flag passes, then it does a full run to get max achievable QPS before and after the end of cracking (default False)",
)
parser.add_argument(
    "--clear_results",
    action="store_true",
    default=False,
    help="If flag passed, then any previous results stored are deleted",
)


# run params
parser.add_argument("--nruns", 
    type=int, 
    default=1, 
    help="number of runs. Default 1 due to cost eestimator, not deterministic, run this multiple time, but not multiple runs"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42, # seed 1 is bad, nprobe 5 and inter=0 is too good, just out of luck, 11 has decent variation between them to be able to check the benefits of training...
    help="seed for kmeans & other random functions",
)
parser.add_argument(
    "--nthreads",
    type=int,
    nargs="+",
    # TODO: Run baselines for 1 and 8 threads but with a timer cut-off to "build-index"
    default=[16, 32, 64, 128], # 1 is too slow for some baselines I wanted numbers
    help="(For CPU only) number of threads, if not set, use the max (default is [16, 32, 64, 128])",
)
parser.add_argument(
    "--runid",
    type=int,
    default=0,
    help="A number you can set to seperate out the storage of runs (default 0)",
)
parser.add_argument(
    "--run_desc",
    type=str,
    default=None,
    help="A short description of the run if you wish. Become part of file name (default None)",
)

# Indexes parameters
parser.add_argument(
    "--index_name",
    type=str,
    nargs="+",
    default=[ "CrackIVF"],
    help="Which index to use. ( default CrackIVF, others: [None currently for this script])",
)
parser.add_argument(
    "--topK",
    type=int,
    default=100,
    help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc",
)
parser.add_argument(
    "--niter",
    type=int,
    nargs="+",
    default=[0],
    help="Number of training iterations for initial KMeans (default [0]). Accepts a list of values.",
)
# TODO: figure out dynamic nprobe when you've cracked, what should the nprobe be at different #centroids, and also region, since some have fewer soem more
parser.add_argument(
    "--nprobe",
    type=int,
    nargs="+",
    default=[5,10,20],
    help="Number of invlists scanned during search (default [5,10,20]). Accepts a list of values.",
)
# TODO: how many centroids you start from should be based on nthreads, dataset, cost of .add() etc. 
#   - eg cost of 1 centroid worse than 100 if you have some threads, becasue .add() parallel across threads
parser.add_argument(
    "--nlist",
    type=int,
    nargs="+",
    default=[100],
    help="Number of centroids/inverted lists to start from (defaults [100])",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None, # 1 is too slow for some baselines I wanted numbers
    help="Batch size for query exectutino, (default None, will set batch size to nthreads)",
)
parser.add_argument(
    "--detailed",
    action="store_true",
    default=False,
    help="If to store the super detailed df where it has all individual crack timings (default True)",
)

# Ablation parameters
parser.add_argument('--alpha', type=float, default=0.5, help='Fraction of total time allowed for build operations (default: 0.5)')
parser.add_argument('--cv_max', type=float, default=2, help='Max coefficient of variation for local imbalance before refine (default: 2)')
parser.add_argument('--spread_max', type=float, default=10, help='Max local spread before refine (default: 10)')
parser.add_argument('--size_prcnt_low', type=float, default=10, help='Percentile for small cluster threshold (default: 10)')
parser.add_argument('--size_prcnt_high', type=float, default=90, help='Percentile for large cluster threshold (default: 90)')
parser.add_argument('--min_pts', type=int, default=2, help='Minimum points allowed after a crack (default: 2)')
parser.add_argument('--pts_crack_thr', type=int, default=64, help='Threshold points per centroid for cracking (default: 64)')


args = parser.parse_args()

#################################################################
# Setup
#################################################################

dbname = args.dbname
index_name_list = args.index_name
skew = args.skew

topK = args.topK
n_iter_list = args.niter  # list
nlist_list = args.nlist # list
nprobe_list = args.nprobe  # list
nruns = args.nruns
starting_seed = args.seed
alpha = args.alpha
cv_max = args.cv_max
spread_max = args.spread_max
size_prcnt_low = args.size_prcnt_low
size_prcnt_high = args.size_prcnt_high
min_pts = args.min_pts
pts_crack_thr = args.pts_crack_thr


if args.batch_size is None:
    batch_size = args.nthreads
    N_CORES = args.nthreads[0]  # Set your desired number of cores
    os.environ["OMP_NUM_THREADS"] = str(N_CORES)  # Limits OpenMP-based parallelism
    os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)  # Limits OpenBLAS (NumPy, SciPy)
    os.environ["MKL_NUM_THREADS"] = str(N_CORES)  # Limits Intel MKL (NumPy, SciPy)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_CORES)  # Limits macOS Accelerate
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_CORES)  # Limits NumExpr computations
else:
    batch_size = args.batch_size
    
result_dir = os.path.join( args.result_path, f"{dbname}_{skew}_OURS")
if not os.path.isdir(result_dir):
    print("%s does not exist, creating it" % result_dir)
    os.mkdir(result_dir)



#################################################################
# Load Dataset
#################################################################

xb, xq, gt, metric = load_dataset(dbname=args.dbname)
original_num_queries = xq.shape[0]
# trimit = 1
# xq = xq[0:trimit,:]
# gt = gt[0:trimit,:]
print(f"Dataset [{args.dbname}] Loaded.")

### Add skew, duplicate up to target queries 
#       - TODO: add these to vasili_helpers? 

# Optional, if skew != default : add some skew to the query set
if skew == "default":
    # typically default == uniform workload. Default uses the queries provided with a particular dataset ~100% clusters accessed
    pass
else:
    if skew == "max":
        # TODO: sample a single point and do pertrubations around it to generate other queries around that point ~1-5% clusters accessed
        pass
    if skew == "high":
    # sample points from a single cluster ~10-15% clusters clusters accessed
        ncents = 1000
        _, xq, gt, _ = get_skewed_dataset(
            xb, 
            skew_on_centroid = int(args.seed % ncents), 
            nlist=ncents, 
            compute_GT = True, 
            nearest_cents_to_include=1,
            seed=args.seed,
            print_skew=True,
            plt_skew_nprobe=20,
        )
    elif skew == "medium":
        # TODO: sample from many cluster ( can be multiple hot regions ) ~50% accessed
        pass
    elif skew == "low":
        # TODO: ~80% accessed
        pass

if args.target_queries > 0:
    # if not enough queries, increase by duplicating this query set until we reach target number of queries (target_queries)
    # factor = 1 means no duplication will take place
    factor = int(args.target_queries / xq.shape[0]) + 1 # duplicate until you have more than target num queires
    if factor != 1:
        print(f"duplicating query set {xq.shape} by {factor=}")
        xq, gt = increase_queries_by_duplicates(xq, gt, factor)
        print(f"after: query set {xq.shape}")
    xq = xq[:args.target_queries, :]
    gt = gt[:args.target_queries, :]

nq, d = xq.shape
assert gt.shape[0] == nq
# print(f"nq={nq}")#DEBUG
# Create the command string
command = "python run_ours.py"
for arg, value in vars(args).items():
    if isinstance(value, list):
        # Join list values with space
        value = " ".join(map(str, value))
    if isinstance(value, bool):
        # Add flags for boolean arguments
        if value:
            command += f" --{arg}"
    else:
        # Add other arguments with their values
        command += f" --{arg} {value}"

# Print the command
print(command)

# Define both summary and detailed baseline CSVs
base_summary_fname = "summary_results"
if args.detailed:
    base_detailed_fname = "super_detailed_results" # lol the name
else:
    base_detailed_fname = "detailed_results"

if args.run_desc is not None:
    base_summary_fname += f"_{args.run_desc}"
    base_detailed_fname += f"_{args.run_desc}"
if args.runid is not None and args.runid != 0:
    base_summary_fname += f"_{args.runid}"
    base_detailed_fname += f"_{args.runid}"
summary_results_path = os.path.join(result_dir, f"{base_summary_fname}.csv")
if args.get_qps: qps_summary_results_path = os.path.join(result_dir, f"qps_{base_summary_fname}.csv") # this is only for args
detailed_results_path = os.path.join(result_dir, f"{base_detailed_fname}.csv")


summary_columns = [
    "dataset_name","skew","seed","nruns","index_name","nlist","cracks_history",
    "nprobe","niter","add_time","train_time","total_build_time","cracking_overhead_time","total_search_time",
    "total_time","10th_search_time_ms","25th_search_time_ms","median_search_time_ms",
    "mean_search_time_ms","75th_search_time_ms","90th_search_time_ms","99th_search_time_ms",
    "std_search_time_ms","CV_search_time_ms","QPS","recall_1","recall_10","recall_100",
    "nthreads","batch_size","total_num_queries","alpha","min_pts","pts_crack_thr","cv_max",
    "spread_max","size_prcnt_low","size_prcnt_high","run_date","cmd"
]

#NOTE: dont do this one but it's commented out if you wish, look later in the code
super_detailed_columns = [
    'batch_id', 'index_name', 'batch_size', 'qid', 'nthreads', 'curr_nlist',
    "nlist","niter", 'seed', 'nprobe', 'cummulative_time_ms_mean', 'total_ms_mean',
    'overhead_ms_mean', 'search_time_ms_mean', 'crack_time_ms_mean',
    'refine_time_ms_mean', 'get_local_region_time_ms_mean',
    'reorg_time_ms_mean', 'inner_point_assignments_ms_mean',
    'update_invlists_ms_mean', 'metadata_tracking_time_mean', 'nruns',
    "recall_1_perQ","recall_10_perQ","recall_100_perQ", "cum_dist_comp"
]

detailed_columns =[
    "batch_id","index_name","batch_size","qid","nthreads","curr_nlist","nlist","niter","seed","nprobe",
    "cummulative_time_ms_mean","total_ms_mean","overhead_ms_mean","search_time_ms_mean"
]

summary_df = handle_baseline_csv(summary_results_path, summary_columns, args.clear_results)
if args.get_qps:  qps_summary_df = handle_baseline_csv(qps_summary_results_path, summary_columns, args.clear_results)
if args.detailed:
    detailed_df = handle_baseline_csv(detailed_results_path, super_detailed_columns, args.clear_results)
else:
    detailed_df = handle_baseline_csv(detailed_results_path, detailed_columns, args.clear_results)


# Optional: Display settings for easier debugging
pd.set_option("display.expand_frame_repr", False)


#################################################################
# BUILD / SEARCH
#################################################################

# warm-up so that .train() doesn't take long even though niter=0
print("Warmup:")
warmup_index, train_time, add_time = train_ivfflat(
                xb,
                # nlist=1_000, # NOTE: if you plot skew set this manually to num cracks you end with so that it comparable
                # nlist=4000,
                km_n_iter=0,
                # km_n_iter=10, # NOTE: if you plot skew should be trained
                km_max_pts=256,
                seed=42,
                store_dir=None,  # if you want to store the index
                verbose=False,
                metric=metric,
                # store=args.store,
            )
            
if args.plot:
        #BUG: The skew plot on TEXT2IMAGE there's no way it's correct, apple to oranges comparison see total frequency,
        _ = get_centroid_ids_per_query(
                index=warmup_index, # D(x)
                # combined_index, # D(x), Q(x) 
                queries=xq[:original_num_queries], #NOTE: make sure you run up to original_num_queries, no duplicates there's no reson, same w/ the AFTER plot
                # queries=xq,
                nprobe=1, 
                plot_hist=True,
                sorted_order=True,
                dataset_name = f"(before) dataset access skew",
                save_plot = False, # no need to save here, we save at the end
                save_path = result_dir
            )

print("===================")        
print("STARTING BENCHMARK:")
print("===================")        
qps_recall_results = defaultdict(list) # across all datasets
if args._get_args: 
    finished_qps_start = True
    finished_qps_end = False

for nthreads in args.nthreads:
    assert nthreads != None
    
    if args.batch_size is None:
        batch_size = nthreads
    else:
        batch_size = args.batch_size

    faiss.omp_set_num_threads(nthreads)
    print(f"Run with : {nthreads=} and {batch_size=}...")
    
    for index_name in index_name_list:
        #### Run CrackIVF baseline
        if index_name == "CrackIVF":
            refine_nprobe = 100 
            print()
            print(f"--- {index_name} ---")
            for nlist in nlist_list:
                for n_iter in n_iter_list:
                    # TODO: do I need to iterate over nprobe or have dynamic nprobe here?
                    
                    # NOTE: This is for QPS only
                    if args.get_qps and not finished_qps_start:
                        # NOTE: we might have duplicated nq to crack longer etc, no need to get QPS from entire thing, just use each query once 
                        nq_qps = original_num_queries
                        gt_qps = gt[:nq_qps]
                        # do a full run at start of cracking 
                        index_name_temp = "CrackIVF-START"
                        print(f"(QPS RUN- START): Running {index_name_temp} -as is- at START, to get current QPS-Recall performance.")
                        nprobes_to_scan = [1,2,4,8,16,32,64,128,256,512,1024]
                        # exatly same index, same seed etc.
                        index_to_crack = CrackIVF(
                                nlist=nlist, 
                                niter=n_iter, 
                                max_pts=256, # faiss default
                                seed=starting_seed,
                                metric=metric,
                                dbname=dbname,
                                nthreads=nthreads,
                                alpha=alpha,
                                cv_max=cv_max,
                                spread_max=spread_max,
                                size_prcnt_low=size_prcnt_low,
                                size_prcnt_high=size_prcnt_high,
                                min_pts=min_pts,
                                pts_crack_thr=pts_crack_thr,
                            )
                        index_to_crack.verbose = False # avoid printing the training details each time?
                        _, _ = index_to_crack.add(xb)
                        index_to_crack.converged = True # fix index at start

                        finished_at_least_one_run = False
                        for nprobe in nprobes_to_scan:
                            if finished_at_least_one_run and recall_10_list[-1] > 0.99:
                                # if previous nprobe value achieved recall > .99 no need to incerase nprobe further
                                continue #skip, mig
                            index_to_crack.nprobe = nprobe # set the nprobe 
                            
                            latency_ms_per_batch = []
                            # lists across runs
                            qps_across_runs = []
                            recall_1_list, recall_10_list, recall_100_list = [], [], []

                            # multiple runs per nprobe to handle H/W variance
                            # THIS IS COPIED FROM BASELINES, SO THAT WE MEASURE QPS THE SAME WAY AS BASLINES!
                            for run in range(nruns):
                                metric_type = "Angular or Inner Product" if index_to_crack.index.metric_type == 0 else "L2" if index_to_crack.index.metric_type == 1 else "Unknown"
                                print(f"{run=}, {n_iter=}, {batch_size=}, {nprobe=}, Metric: {metric_type}")

                                I = np.empty((nq_qps, topK), dtype="int32")
                                D = np.empty((nq_qps, topK), dtype="float32")
                                latency_per_batch = []
                                
                                i0 = 0
                                t0 = time.perf_counter()
                                while i0 < nq_qps:
                                    i1 = min(i0 + batch_size, nq_qps)

                                    t_batch_start = time.perf_counter()
                                    Di, Ii, _ = index_to_crack.search(xq[i0:i1], topK)
                                    t_batch_end = time.perf_counter()

                                    I[i0:i1] = Ii
                                    D[i0:i1] = Di

                                    latency_per_batch.append(t_batch_end - t_batch_start)
                                    i0 = i1
                                    if index_to_crack.total_batches == 1 or index_to_crack.total_batches_since_last_crack == 1:
                                        print(f"After Crack {index_to_crack.index.nlist} stats:")
                                        index_to_crack.index.invlists.print_stats() # print stats after each crack


                                t1 = time.perf_counter()
                                time_entire_loop = (t1 - t0)
                                
                                # Get QPS, Response Times, Recalls and append to lists
                                latency_per_batch = np.array(latency_per_batch)
                                latency_per_batch_ms = latency_per_batch * 1000
                                search_time_prcnt_10_ms = np.percentile(latency_per_batch_ms, 10)
                                search_time_prcnt_25_ms = np.percentile(latency_per_batch_ms, 25)
                                search_time_prcnt_75_ms = np.percentile(latency_per_batch_ms, 75)
                                search_time_prcnt_90_ms = np.percentile(latency_per_batch_ms, 90)
                                search_time_prcnt_99_ms = np.percentile(latency_per_batch_ms, 99)
                                median_search_time_ms = np.median(latency_per_batch_ms)
                                mean_search_time_ms = np.mean(latency_per_batch_ms)
                                # variance measures
                                std_search_time_ms = np.std(latency_per_batch_ms)
                                CV_search_time_ms = (std_search_time_ms / mean_search_time_ms) * 100
                                
                                total_search_time = sum(latency_per_batch)
                                qps = nq_qps / total_search_time

                                qps_across_runs.append(qps)
                                recall_1_list.append(compute_recall(I, gt_qps, 1))
                                recall_10_list.append(compute_recall(I, gt_qps, 10))
                                recall_100_list.append(compute_recall(I, gt_qps, 100))
                                print_recall(I, gt_qps)


                                latency_ms_per_batch_this_run = np.array(latency_per_batch) * 1000
                                if (
                                    len(latency_ms_per_batch_this_run) > 2
                                ):  # remove first and last batch latency
                                    latency_ms_per_batch_this_run = latency_ms_per_batch_this_run[
                                        1:-1
                                    ]
                                latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                                print(
                                    "Median batch latency: {:.3f} ms".format(
                                        np.median(latency_ms_per_batch_this_run)
                                    ) + " | " +
                                    "Mean batch latency: {:.3f} ms".format(
                                        np.mean(latency_ms_per_batch_this_run)
                                    ) + " | " +
                                    "99th percentile batch latency: {:.3f} ms".format(
                                        np.percentile(latency_ms_per_batch_this_run, 99)
                                    )
                                )
                                print(f"{index_name_temp} imbalance = {index_to_crack.index.invlists.imbalance_factor()} & {time_entire_loop=} sec\n")
                            
                            finished_at_least_one_run = True
                            median_qps_across_runs = np.median(qps_across_runs)
                            recall_1 = np.median(recall_1_list)
                            recall_10 = np.median(recall_10_list)
                            recall_100 = np.median(recall_100_list)
                            assert len(set(recall_1_list)) == 1, "Recall@1 values are not the same across runs"
                            assert len(set(recall_10_list)) == 1, "Recall@10 values are not the same across runs"
                            assert len(set(recall_100_list)) == 1, "Recall@100 values are not the same across runs"
                            
                            # finished this nprobe, add a row
                            qps_summary_row_to_add = {
                                "dataset_name": dbname,
                                "skew": skew,
                                "seed": starting_seed,
                                "nruns": args.nruns,
                                "index_name": index_name_temp, # QPS INDEX NAME!
                                "nlist": nlist,  # (starting nlist one)
                                "cracks_history": f'{[nlist]}',# no cracks
                                "nprobe": nprobe, # added later TODO: nprobe history if it varies by num cracks?
                                "niter": n_iter,
                                "add_time": None,
                                "train_time": None,
                                "cracking_overhead_time": None,
                                "total_build_time": None, # train + add + sum of all overhead and divide by 1k to turn into seconds
                                "total_search_time": np.mean(total_search_time), # sum(search)
                                # TODO: if you do time entire loop, do the same in the baselines
                                "total_time": None, # first mean is across runs per batch, then sum across all batches, then divide by 1k to turn into sec
                                "10th_search_time_ms": search_time_prcnt_10_ms,
                                "25th_search_time_ms":search_time_prcnt_25_ms,
                                "median_search_time_ms": median_search_time_ms,
                                "mean_search_time_ms": mean_search_time_ms,
                                "75th_search_time_ms": search_time_prcnt_75_ms,
                                "90th_search_time_ms": search_time_prcnt_90_ms,
                                "99th_search_time_ms": search_time_prcnt_99_ms,
                                "std_search_time_ms":std_search_time_ms,
                                "CV_search_time_ms":CV_search_time_ms,
                                "QPS": median_qps_across_runs, # total_num_queries / total_search_time
                                "recall_1": recall_1,
                                "recall_10": recall_10,
                                "recall_100": recall_100,
                                "nthreads": nthreads,
                                "batch_size": batch_size,
                                "total_num_queries": nq_qps, 
                                "alpha": alpha,
                                "min_pts": min_pts,
                                "pts_crack_thr": pts_crack_thr,
                                "cv_max": cv_max,
                                "spread_max": spread_max,
                                "size_prcnt_low": size_prcnt_low,
                                "size_prcnt_high": size_prcnt_high,
                                "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
                                "cmd": command,
                            }
                            # store/append csv:
                            qps_summary_df = add_rows_to_csv(qps_summary_row_to_add, qps_summary_df, qps_summary_results_path, is_single_row=True, store=args.store)
                    finished_qps_start = True

                    ## ######## ----  ######## #### ######## ----  ######## #### ######## ----  ######## ##
                    # ##ACTUAL RUN:
                    ## ######## ----  ######## #### ######## ----  ######## #### ######## ----  ######## ##

                    print(f"--- {index_name} ---")
                    print("(CRACKING RUN): Start normal run w/ Cracking. This changes the index as more queries arrive.")
                    for nprobe in nprobe_list:
                        # if nprobe >= 0.75*nlist:
                        #     continue
                        latency_ms_per_batch = [] # latency including cracking overhead 
                        search_latency_ms_per_batch = [] # latency for how soon search results are available and can be sent to the user
                        # lists across runs
                        qps_across_runs = []
                        recall_1_list, recall_10_list, recall_100_list = [], [], []

                        # multiple runs per nprobe to handle H/W variance
                        detailed_rows_to_add = [] # across runs
                        train_times, add_times,total_search_time = [],[], []
                        
                        
                        for run in range(nruns):
                            I = np.empty((nq, topK), dtype="int32")
                            D = np.empty((nq, topK), dtype="float32")
                            latency_per_batch = []
                            search_latency_per_batch = []
                            
                            print(f"Build for {run=}, {n_iter=}, {batch_size=}, {nprobe=}, {refine_nprobe=}")
                            
                            ### BUILD INDEX from scratch for each run (slower but you make sure you start from same point and crack again because cracks applied in place)
                            print()
                            # Build index for nlist niter combination
                            index_to_crack = CrackIVF(
                                nlist=nlist, 
                                niter=n_iter, 
                                max_pts=256, # faiss default
                                seed=starting_seed,
                                metric=metric,
                                dbname=dbname,
                                nthreads=nthreads,
                                alpha=alpha,
                                cv_max=cv_max,
                                spread_max=spread_max,
                                size_prcnt_low=size_prcnt_low,
                                size_prcnt_high=size_prcnt_high,
                                min_pts=min_pts,
                                pts_crack_thr=pts_crack_thr,
                            )
                            index_to_crack.verbose = False # avoid printing the training details each time?
                            train_time, add_time = index_to_crack.add(xb)
                            train_times.append(train_time)
                            add_times.append(add_time)
                            index_to_crack.nprobe = nprobe # set the nprobe 
                            index_to_crack.refine_nprobe = refine_nprobe # set the local refine region nprobe
                            metric_type = "Inner Product" if index_to_crack.index.metric_type == 0 else "L2" if index_to_crack.index.metric_type == 1 else "Unknown"
                            print(f"Search for {run=}, {n_iter=}, {batch_size=}, {nprobe=}, {refine_nprobe=}, Metric: {metric_type}")


                            # TODO: figure out if you store qps of each run seperately?
                            #       - NOTE: each run should be deterministic on H/W variance so keep mean or medians...
                            #       - 
                            cracks_history = [index_to_crack.index.nlist]

                            # you've already paid some train/add cost before first batch
                            cummulative_time_ms = train_time * 1000 + add_time * 1000
                            i0 = 0
                            batch_num = 0
                            t0 = time.perf_counter()

                            while i0 < nq:
                                qid_batch_start = i0
                                
                                i1 = min(i0 + batch_size, nq)

                                t_batch_start = time.perf_counter()
                                Di, Ii, timings = index_to_crack.search(xq[i0:i1], topK)
                                t_batch_end = time.perf_counter()

                                I[i0:i1] = Ii
                                D[i0:i1] = Di

                                latency_per_batch.append(t_batch_end - t_batch_start)
                                cummulative_time_ms += timings["total"][0] * 1000 # total
                                search_latency_per_batch.append(timings["search"][0]) # search time
                                overhead_curr_batch_ms = (timings["total"][0] - timings["search"][0]) * 1000 # overhead is everything not search
                                curr_num_cracks = index_to_crack.index.nlist
                                if curr_num_cracks != cracks_history[-1]:
                                    cracks_history.append(curr_num_cracks)
                                
                                
                                # print(f"{Ii.shape=} , {gt.shape=}, {i0=}, {i1=},{gt[i0:i1].shape=}") # DEBUG
                                # NOTE: these (+the row updates etc) slow down the time_entire_loop, so total time is actually less than time_entire_loop
                                recall_1_perQ = compute_recall(Ii, gt[i0:i1,:], k=1)
                                recall_10_perQ = compute_recall(Ii, gt[i0:i1,:], k=10)
                                recall_100_perQ = compute_recall(Ii, gt[i0:i1,:], k=100)

                                i0 = i1
                                batch_num += 1
                                
                                
                                detailed_row = {
                                    "index_name": index_name,
                                    "qid": qid_batch_start,
                                    "batch_id": batch_num,
                                    "batch_size": batch_size,
                                    "nthreads": nthreads,
                                    "seed": starting_seed,
                                    "curr_nlist": curr_num_cracks,
                                    "nlist": nlist,
                                    "niter": n_iter,
                                    "cummulative_time_ms": cummulative_time_ms,  # cumulative of total time
                                    "nprobe": nprobe,  # current nprobe (may vary in more advanced implementation)
                                    "total_ms": timings["total"][0] * 1000,
                                    "overhead_ms": overhead_curr_batch_ms,
                                    "search_time_ms": timings["search"][0] * 1000
                                }

                                # Add detailed columns if args.detailed is True
                                if args.detailed:
                                    detailed_row.update({
                                        "crack_time_ms": timings["crack"][0] * 1000 if len(timings["crack"]) > 0 else 0,
                                        "refine_time_ms": timings["refine"][0] * 1000 if len(timings["refine"]) > 0 else 0,
                                        "get_local_region_time_ms": timings["get_local_region"][0] * 1000 if len(timings["get_local_region"]) > 0 else 0,
                                        "reorg_time_ms": timings["reorg"][0] * 1000 if len(timings["reorg"]) > 0 else 0,
                                        "inner_point_assignments_ms": timings["inner_point_assignments"][0] * 1000 if len(timings["inner_point_assignments"]) > 0 else 0,
                                        "update_invlists_ms": timings["update_invlists"][0] * 1000 if len(timings["update_invlists"]) > 0 else 0,
                                        "metadata_tracking_time": timings["metadata_tracking"][0] * 1000 if len(timings["metadata_tracking"]) > 0 else 0,
                                        "nruns": nruns,
                                        "run_id": run,  # aggregate across them
                                        "recall_1_perQ": recall_1_perQ,
                                        "recall_10_perQ": recall_10_perQ,
                                        "recall_100_perQ": recall_100_perQ,
                                        "cum_dist_comp": index_to_crack.cum_dist_comp,
                                    })

                                detailed_rows_to_add.append(detailed_row)

                            t1 = time.perf_counter()
                            time_entire_loop = (t1 - t0) # am I also computing recalls and gathering data in loop, so trust the inner loop timers the most
                            
                            # Get QPS, Response Times, Recalls and append to lists based on search time
                            search_latency_per_batch = np.array(search_latency_per_batch)
                            search_latency_per_batch_ms = search_latency_per_batch * 1000
                            search_time_prcnt_10_ms = np.percentile(search_latency_per_batch_ms, 10)
                            search_time_prcnt_25_ms = np.percentile(search_latency_per_batch_ms, 25)
                            search_time_prcnt_75_ms = np.percentile(search_latency_per_batch_ms, 75)
                            search_time_prcnt_90_ms = np.percentile(search_latency_per_batch_ms, 90)
                            search_time_prcnt_99_ms = np.percentile(search_latency_per_batch_ms, 99)
                            median_search_time_ms = np.median(search_latency_per_batch_ms)
                            mean_search_time_ms = np.mean(search_latency_per_batch_ms)
                            # variance measures
                            std_search_time_ms = np.std(search_latency_per_batch_ms)
                            CV_search_time_ms = (std_search_time_ms / mean_search_time_ms) * 100
                            
                            # QPS should based on search time alone, since 
                            # 1) on baselines we don't include build to QPS
                            # 2) search results available after .search(), .cracking can be handles in background if atomic
                            #   - NOTE: see cummulative time plot to include total search + cracking cost results, not QPS
                            # TODO: should I also keep track of QPS at different "states" of the CrackIVF?
                            #       - eg QPS when it has 100 centroids, when 1k, when 2k, at the end etc.
                            #           - how to decide when these are?
                            
                            # total_search_time = sum(latency_per_batch) # search + crack overhead
                            total_search_time = sum(search_latency_per_batch) # search only
                            qps = nq / total_search_time
                            qps_across_runs.append(qps)

                            recall_1_list.append(compute_recall(I, gt, 1))
                            recall_10_list.append(compute_recall(I, gt, 10))
                            recall_100_list.append(compute_recall(I, gt, 100))
                            print_recall(I, gt)

                            # recomputing some value but ok
                            latency_ms_per_batch_this_run = np.array(latency_per_batch) * 1000
                            if (
                                len(latency_ms_per_batch_this_run) > 2
                            ):  # remove first and last batch latency
                                latency_ms_per_batch_this_run = latency_ms_per_batch_this_run[
                                    1:-1
                                ]
                            latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                            print(
                                "Median batch latency: {:.3f} ms".format(
                                    np.median(latency_ms_per_batch_this_run)
                                ) + " | " +
                                "Mean batch latency: {:.3f} ms".format(
                                    np.mean(latency_ms_per_batch_this_run)
                                ) + " | " +
                                "99th percentile batch latency: {:.3f} ms".format(
                                    np.percentile(latency_ms_per_batch_this_run, 99)
                                )
                            )

                            search_latency_per_batch_this_run_ms = np.array(search_latency_per_batch) * 1000
                            if (
                                len(search_latency_per_batch_this_run_ms) > 2
                            ):  # remove first and last batch latency
                                search_latency_per_batch_this_run = search_latency_per_batch_this_run_ms[
                                    1:-1
                                ]
                            search_latency_ms_per_batch.extend(search_latency_per_batch_this_run_ms)
                            print(
                                "Median search latency: {:.3f} ms".format(
                                    np.median(search_latency_per_batch_this_run_ms)
                                ) + " | " +
                                "Mean search latency: {:.3f} ms".format(
                                    np.mean(search_latency_per_batch_this_run_ms)
                                ) + " | " +
                                "99th percentile search latency: {:.3f} ms".format(
                                    np.percentile(search_latency_per_batch_this_run_ms, 99)
                                )
                            )
                            
                            print(f"CrackIVF: imbalance = {index_to_crack.index.invlists.imbalance_factor()} - {time_entire_loop=} (w/log overhead) sec - {cracks_history=}")
                            print(f"CrackIVF print_stats = {index_to_crack.index.invlists.print_stats()=}") 


                        

                        # append rows that is AGGREGATE across runs
                        # append row to dataframe and store .csv
                        # TODO: medians for cracks etc also? CrackIVF across runs should be deterministic.. (when I crack, when I don't etc)
                        #       - TODO: make sure cracking decisions are deterministic so that different runs are to remove H/W variance
                        median_qps_across_runs = np.median(qps_across_runs)
                        recall_1 = np.median(recall_1_list)
                        recall_10 = np.median(recall_10_list)
                        recall_100 = np.median(recall_100_list)
                        assert len(set(recall_1_list)) == 1, "Recall@1 values are not the same across runs"
                        assert len(set(recall_10_list)) == 1, "Recall@10 values are not the same across runs"
                        assert len(set(recall_100_list)) == 1, "Recall@100 values are not the same across runs"

                        # Aggregate metrics across runs for each batch
                        df_all_runs = pd.DataFrame(detailed_rows_to_add)
                        
                        # from pprint import pprint #DEBUG
                        # print("df_all_runs=")#DEBUG
                        # pprint(df_all_runs)#DEBUG

                        # Group by batch_id and compute medians or means
                        # first == Consistent across runs

                        # NOTE: This is SUPER detailed (too much for really long query sequencies)
                        # SOS NOTE:
                        # ```
                        # To average across runs and remove H/W variance, however small, the algorithm needs to be deterministic.
                        # If you are using the cost estimator, which then takes into account timing variations, to decide WHEN to crack
                        # then IT IS NOT DETERMINISTIC <----
                        # - because we're using timers to decide algorithm behaviour
                        # - do a single run at a time, but then MULTIPLE TIMES to make sure consistent plots/behaviour..
                        #   - just don't average it in single script like here
                        # ```
                        if args.detailed:
                            df_agg_across_runs = df_all_runs.groupby("batch_id").agg({
                                "index_name":"first",
                                "batch_size": "first",  
                                "qid": "first",
                                "nthreads": "first",
                                "curr_nlist": "first",
                                "nlist":"first",
                                "niter": "first",
                                "seed": "first",
                                "nprobe": "first",
                                "cummulative_time_ms": ["mean"], # , ["mean", "median","std"]
                                "total_ms": ["mean"],
                                "overhead_ms": ["mean"],
                                "search_time_ms":["mean"],
                                "crack_time_ms": ["mean"],
                                "refine_time_ms": ["mean"],
                                "get_local_region_time_ms": ["mean"],
                                "reorg_time_ms": ["mean"],
                                "inner_point_assignments_ms":["mean"],
                                "update_invlists_ms": ["mean"],
                                "metadata_tracking_time": ["mean"],
                                "nruns": "first",
                                "recall_1_perQ": "first", # if deterministic: fixed crack schedule not cost est, then first==mean etc, 
                                "recall_10_perQ": "first",
                                "recall_100_perQ": "first",
                                "cum_dist_comp": "first",
                            }).reset_index()

                            # Rename columns for clarity
                            df_agg_across_runs.columns = [
                                "_".join(col).strip("_") if col[1] in {"mean", "median", "std"} else col[0]
                                for col in df_agg_across_runs.columns
                            ]
                        else:
                            # less detailed
                            df_agg_across_runs = df_all_runs.groupby("batch_id").agg({
                                "index_name":"first",
                                "batch_size": "first",  
                                "qid": "first",
                                "nthreads": "first",
                                "curr_nlist": "first",
                                "nlist":"first",
                                "niter": "first",
                                "seed": "first",
                                "nprobe": "first",
                                "cummulative_time_ms": "mean",
                                "total_ms": "mean",
                                "overhead_ms": "mean",
                                "search_time_ms": "mean",
                            }).reset_index()

                            df_agg_across_runs = df_agg_across_runs.rename(
                                columns={
                                        "cummulative_time_ms": "cummulative_time_ms_mean",
                                        "total_ms": "total_ms_mean",
                                        "overhead_ms": "overhead_ms_mean",
                                        "search_time_ms": "search_time_ms_mean",
                                    }
                                )
                        # NOTE :
                        # - like in baslines some of the values here, like search time means etc are actually only from the final run
                        # - but to be fast right now, just leave it, for CrackIVF it doesn't matter, i don't use those, the cummulative plot is accuracte from the detailed df
                        # - and the QPS is also accurate (median across runs) as well as recalls and total build times (also I take means as you see)
                        # - so from CrackIVF as accurate as you can get, TODO: do it for baselines as well if I see huge discrepancy ( unless I get detaild DF for select basleines in which case also ok)
                        summary_row_to_add = {
                            "dataset_name": dbname,
                            "skew": skew,
                            "seed": starting_seed,
                            "nruns": args.nruns,
                            "index_name": index_name,
                            "nlist": nlist,  # (starting nlist one)
                            "cracks_history": cracks_history,
                            "nprobe": nprobe, # added later TODO: nprobe history if it varies by num cracks?
                            "niter": n_iter,
                            "add_time": np.mean(add_times),
                            "train_time": np.mean(train_times),
                            "cracking_overhead_time": (df_agg_across_runs.overhead_ms_mean.sum()/1000),
                            "total_build_time": np.mean(add_times) + np.mean(train_times) + (df_agg_across_runs.overhead_ms_mean.sum()/1000), # train + add + sum of all overhead and divide by 1k to turn into seconds
                            "total_search_time": np.mean(total_search_time), # sum(search)
                            # TODO: if you do time entire loop, do the same in the baselines
                            "total_time": df_agg_across_runs.total_ms_mean.sum() /1000, # first mean is across runs per batch, then sum across all batches, then divide by 1k to turn into sec
                            "10th_search_time_ms": search_time_prcnt_10_ms,
                            "25th_search_time_ms":search_time_prcnt_25_ms,
                            "median_search_time_ms": median_search_time_ms,
                            "mean_search_time_ms": mean_search_time_ms,
                            "75th_search_time_ms": search_time_prcnt_75_ms,
                            "90th_search_time_ms": search_time_prcnt_90_ms,
                            "99th_search_time_ms": search_time_prcnt_99_ms,
                            "std_search_time_ms":std_search_time_ms,
                            "CV_search_time_ms":CV_search_time_ms,
                            "QPS": median_qps_across_runs, # total_num_queries / total_search_time
                            "recall_1": recall_1,
                            "recall_10": recall_10,
                            "recall_100": recall_100,
                            "nthreads": nthreads,
                            "batch_size": batch_size,
                            "total_num_queries": nq, 
                            "alpha": alpha,
                            "min_pts": min_pts,
                            "pts_crack_thr": pts_crack_thr,
                            "cv_max": cv_max,
                            "spread_max": spread_max,
                            "size_prcnt_low": size_prcnt_low,
                            "size_prcnt_high": size_prcnt_high,
                            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
                            "cmd": command,
                        }
                        
                        # store/append csv:
                        summary_df = add_rows_to_csv(summary_row_to_add, summary_df, summary_results_path, is_single_row=True, store=args.store)
                        detailed_df = add_rows_to_csv(df_agg_across_runs.to_dict(orient='records'), detailed_df, detailed_results_path, is_single_row=False, store=args.store)



                ## ######## ----  ######## #### ######## ----  ######## #### ######## ----  ######## ##
                # ##FINISHED CRACKING RUN:
                ## ######## ----  ######## #### ######## ----  ######## #### ######## ----  ######## ##
                    # NOTE: This is for QPS only
                    if args.get_qps and not finished_qps_end:
                        # NOTE: we might have duplicated nq to crack longer etc, no need to get QPS from entire thing, just use each query once 
                        # BUG : Does not work when I addd skew, need to think about it
                        nq_qps = original_num_queries
                        gt_qps = gt[:nq_qps]


                        # do a full run at start of cracking 
                        index_name_temp = "CrackIVF-END"
                        print(f"(QPS RUN- END): Running {index_name_temp} -as is- at END, to get current QPS-Recall performance.")
                        nprobes_to_scan = [1,2,4,8,16,32,64,128,256,512,1024]
                        # we already have index_to_crack and it was cracked in cracking run.
                        index_to_crack.converged = True # fix index at start

                        finished_at_least_one_run = False
                        for nprobe in nprobes_to_scan:
                            # if finished_at_least_one_run and recall_10_list[-1] > 0.99:
                            if finished_at_least_one_run and recall_10_list[-1] > 0.96: # just to speed up for DEBUG
                                # if previous nprobe value achieved recall > .99 no need to incerase nprobe further
                                continue #skip, mig
                            index_to_crack.nprobe = nprobe # set the nprobe 
                            
                            latency_ms_per_batch = []
                            # lists across runs
                            qps_across_runs = []
                            recall_1_list, recall_10_list, recall_100_list = [], [], []

                            # multiple runs per nprobe to handle H/W variance
                            # THIS IS COPIED FROM BASELINES, SO THAT WE MEASURE QPS THE SAME WAY AS BASLINES!
                            for run in range(nruns):
                                metric_type = "Inner Product" if index_to_crack.index.metric_type == 0 else "L2" if index_to_crack.index.metric_type == 1 else "Unknown"
                                print(f"{run=}, {n_iter=}, {batch_size=}, {nprobe=}, Metric: {metric_type}, {index_to_crack.index.nlist=}")

                                I = np.empty((nq_qps, topK), dtype="int32")
                                D = np.empty((nq_qps, topK), dtype="float32")
                                latency_per_batch = []
                                
                                i0 = 0
                                t0 = time.perf_counter()
                                while i0 < nq_qps:
                                    i1 = min(i0 + batch_size, nq_qps)

                                    t_batch_start = time.perf_counter()
                                    Di, Ii, _ = index_to_crack.search(xq[i0:i1], topK)
                                    t_batch_end = time.perf_counter()

                                    I[i0:i1] = Ii
                                    D[i0:i1] = Di

                                    latency_per_batch.append(t_batch_end - t_batch_start)
                                    i0 = i1

                                t1 = time.perf_counter()
                                time_entire_loop = (t1 - t0)
                                
                                # Get QPS, Response Times, Recalls and append to lists
                                latency_per_batch = np.array(latency_per_batch)
                                latency_per_batch_ms = latency_per_batch * 1000
                                search_time_prcnt_10_ms = np.percentile(latency_per_batch_ms, 10)
                                search_time_prcnt_25_ms = np.percentile(latency_per_batch_ms, 25)
                                search_time_prcnt_75_ms = np.percentile(latency_per_batch_ms, 75)
                                search_time_prcnt_90_ms = np.percentile(latency_per_batch_ms, 90)
                                search_time_prcnt_99_ms = np.percentile(latency_per_batch_ms, 99)
                                median_search_time_ms = np.median(latency_per_batch_ms)
                                mean_search_time_ms = np.mean(latency_per_batch_ms)
                                # variance measures
                                std_search_time_ms = np.std(latency_per_batch_ms)
                                CV_search_time_ms = (std_search_time_ms / mean_search_time_ms) * 100
                                
                                total_search_time = sum(latency_per_batch)
                                qps = nq_qps / total_search_time

                                qps_across_runs.append(qps)
                                recall_1_list.append(compute_recall(I, gt_qps, 1))
                                recall_10_list.append(compute_recall(I, gt_qps, 10))
                                recall_100_list.append(compute_recall(I, gt_qps, 100))
                                print_recall(I, gt_qps)


                                latency_ms_per_batch_this_run = np.array(latency_per_batch) * 1000
                                if (
                                    len(latency_ms_per_batch_this_run) > 2
                                ):  # remove first and last batch latency
                                    latency_ms_per_batch_this_run = latency_ms_per_batch_this_run[
                                        1:-1
                                    ]
                                latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                                print(
                                    "Median batch latency: {:.3f} ms".format(
                                        np.median(latency_ms_per_batch_this_run)
                                    ) + " | " +
                                    "Mean batch latency: {:.3f} ms".format(
                                        np.mean(latency_ms_per_batch_this_run)
                                    ) + " | " +
                                    "99th percentile batch latency: {:.3f} ms".format(
                                        np.percentile(latency_ms_per_batch_this_run, 99)
                                    )
                                )
                                print(f"{index_name_temp} imbalance = {index_to_crack.index.invlists.imbalance_factor()} & {time_entire_loop=} sec\n")
                            
                            finished_at_least_one_run = True
                            median_qps_across_runs = np.median(qps_across_runs)
                            recall_1 = np.median(recall_1_list)
                            recall_10 = np.median(recall_10_list)
                            recall_100 = np.median(recall_100_list)
                            assert len(set(recall_1_list)) == 1, "Recall@1 values are not the same across runs"
                            assert len(set(recall_10_list)) == 1, "Recall@10 values are not the same across runs"
                            assert len(set(recall_100_list)) == 1, "Recall@100 values are not the same across runs"
                            
                            # finished this nprobe, add a row
                            qps_summary_row_to_add = {
                                "dataset_name": dbname,
                                "skew": skew,
                                "seed": starting_seed,
                                "nruns": args.nruns,
                                "index_name": index_name_temp, # QPS INDEX NAME!
                                "nlist": index_to_crack.index.nlist,  # (starting nlist one)
                                "cracks_history": cracks_history, # cracks history 
                                "nprobe": nprobe, # added later TODO: nprobe history if it varies by num cracks?
                                "niter": n_iter,
                                "add_time": None,
                                "train_time": None,
                                "cracking_overhead_time": None,
                                "total_build_time": None, # train + add + sum of all overhead and divide by 1k to turn into seconds
                                "total_search_time": np.mean(total_search_time), # sum(search)
                                # TODO: if you do time entire loop, do the same in the baselines
                                "total_time": None, # first mean is across runs per batch, then sum across all batches, then divide by 1k to turn into sec
                                "10th_search_time_ms": search_time_prcnt_10_ms,
                                "25th_search_time_ms":search_time_prcnt_25_ms,
                                "median_search_time_ms": median_search_time_ms,
                                "mean_search_time_ms": mean_search_time_ms,
                                "75th_search_time_ms": search_time_prcnt_75_ms,
                                "90th_search_time_ms": search_time_prcnt_90_ms,
                                "99th_search_time_ms": search_time_prcnt_99_ms,
                                "std_search_time_ms":std_search_time_ms,
                                "CV_search_time_ms":CV_search_time_ms,
                                "QPS": median_qps_across_runs, # total_num_queries / total_search_time
                                "recall_1": recall_1,
                                "recall_10": recall_10,
                                "recall_100": recall_100,
                                "nthreads": nthreads,
                                "batch_size": batch_size,
                                "total_num_queries": nq_qps, 
                                "alpha": alpha,
                                "min_pts": min_pts,
                                "pts_crack_thr": pts_crack_thr,
                                "cv_max": cv_max,
                                "spread_max": spread_max,
                                "size_prcnt_low": size_prcnt_low,
                                "size_prcnt_high": size_prcnt_high,
                                "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
                                "cmd": command,
                            }
                            # store/append csv:
                            qps_summary_df = add_rows_to_csv(qps_summary_row_to_add, qps_summary_df, qps_summary_results_path, is_single_row=True, store=args.store)
                        finished_qps_end = True

                        print(f"{index_name_temp}: imbalance = {index_to_crack.index.invlists.imbalance_factor()} - {time_entire_loop=} (w/log overhead) sec - {cracks_history=}")
                        print(f"{index_name_temp}: print_stats = {index_to_crack.index.invlists.print_stats()=}") 

                    if args.plot:
                            _ = get_centroid_ids_per_query(
                                    index=index_to_crack.index, # D(x)
                                    # combined_index, # D(x), Q(x) 
                                    queries=xq[:original_num_queries],
                                    nprobe=1, 
                                    plot_hist=True,
                                    sorted_order=True,
                                    dataset_name = f"(after) on {index_name} w/ cracks={index_to_crack.index.nlist}",
                                    save_plot = True, # no need to save here, we save at the end
                                    save_path = result_dir
                                )