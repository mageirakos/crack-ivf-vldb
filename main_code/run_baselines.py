from __future__ import print_function
from vasili_helpers import *
import time
import faiss
import os
from datetime import datetime
import pandas as pd
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()

# read & store results & dataset parameters
parser.add_argument(
    "--result_path",
    type=str,
    default="./results/",
    help="path to store results + indexes ",
)
parser.add_argument(
    "--dbname",
    type=str,
    default="SIFT1M",
    help="dataset name, e.g., SIFT1M, (default)",
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
parser.add_argument(
    "--plot",
    action="store_true",
    default=False,
    help="If flag passes, then create and store plots (default False)",
)
# NOTE: avoid clear results it's dangerous
# parser.add_argument(
#     "--clear_results",
#     action="store_true",
#     default=False,
#     help="If flag passed, then any previous results stored are deleted",
# )


# run params
parser.add_argument("--nruns", 
    type=int, 
    default=3, 
    help="number of runs"
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
    type=str,
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
    default=[ "BruteForce", "IVFFlat"],
    help="Which index to use. ( default IVFFlat, other: [IVFFlat, BruteForce])",
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
    default=[0,1,5,10],
    help="Number of training iterations for KMeans (default [1,2,5,10]). Accepts a list of values.",
)
parser.add_argument(
    "--nprobe",
    type=int,
    nargs="+",
    default=[1,2,4,6,8,16,32,64,128,256,512,1024],
    help="Number of invlists scanned during search (default [5,10,20,30,40,50,75]). Accepts a list of values.",
)
parser.add_argument(
    "--nlist",
    type=int,
    nargs="+",
    # TODO: rerun 10k because it was at 1k by accident
    default=[10,100,500,1000,2000,5000,10000,16000],
    help="Number of centroids/inverted lists to train (defaults [10,100,500,1000,2000,5000,1000,16000])",
)
parser.add_argument(
    "--batch_size",
    type=int,
    # TODO: Run baselines for 1 and 8 threads but with a timer cut-off to "build-index"
    default=None, # 1 is too slow for some baselines I wanted numbers
    help="Batch size for query exectutino, (default None, will set batch size to nthreads)",
)
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
    
result_dir = os.path.join( args.result_path, f"{dbname}_{skew}_BASELINES")
if not os.path.isdir(result_dir):
    print("%s does not exist, creating it" % result_dir)
    os.mkdir(result_dir)



#################################################################
# Load Dataset
#################################################################
xb, xq, gt, metric = load_dataset(dbname=args.dbname)
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

# Create the command string
command = "python run_baselines.py"
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

# Define the path to the baseline results CSV
fname = "baseline_results"
if args.run_desc is not None:
    fname += f"_{args.run_desc}"

baseline_results_path = os.path.join(result_dir, f"{fname}.csv")
if args.runid != 0:
    baseline_results_path = os.path.join(result_dir, f"{fname}_{args.runid}.csv")

# Define the expected columns
all_columns = [
    "dataset_name", "dim", "skew", "seed", "nruns", "index_name", "nlist", "nprobe",
    "niter", "add_time", "train_time", "total_build_time", "total_search_time",
    "total_time", "10th_search_time_ms", "25th_search_time_ms", "median_search_time_ms",
    "mean_search_time_ms", "75th_search_time_ms", "90th_search_time_ms", "99th_search_time_ms",
    "std_search_time_ms","CV_search_time_ms","QPS", "recall_1", "recall_10", "recall_100", 
    "nthreads", "batch_size","total_num_queries", "run_date", "cmd"
]

# Check if the CSV file exist
if os.path.exists(baseline_results_path):
    # NOTE: avoid clear results its dangerous
    # if args.clear_results:
    #     os.remove(baseline_results_path)
    #     print(f"Existing CSV file '{baseline_results_path}' deleted. Creating new CSV.")
    #     df = pd.DataFrame(columns=all_columns)
    #     df.to_csv(baseline_results_path, index=False)
    # else:
    
    df = pd.read_csv(baseline_results_path)
    assert all(col in df.columns for col in all_columns), "Column mismatch in CSV"
    print("Baseline results dataframe loaded successfully.")
else:
    df = pd.DataFrame(columns=all_columns)
    df.to_csv(baseline_results_path, index=False)
    print(f"New CSV created at '{baseline_results_path}'")

# Optional: Display settings for easier debugging
pd.set_option("display.expand_frame_repr", False)


#################################################################
# BUILD / SEARCH
#################################################################

# warm-up so that .train() doesn't take long even though niter=0
print("Warmup:")
warmup_index, train_time, add_time = train_ivfflat(
                xb,
                nlist=1000,
                km_n_iter=0,
                km_max_pts=256,
                seed=42,
                store_dir=None,  # if you want to store the index
                verbose=False,
                metric=metric,
                # store=args.store,
            )
            
if args.plot:
        _ = get_centroid_ids_per_query(
                index=warmup_index, # D(x)
                # combined_index, # D(x), Q(x) 
                queries=xq,
                nprobe=20, 
                plot_hist=True,
                sorted_order=True,
                dataset_name = f"dataset_skew-{dbname}-{skew}",
                save_plot = False, # no need to save here, we save at the end
                save_path = result_dir
            )
print("===================")        
print("STARTING BENCHMARK:")
print("===================")        
qps_recall_results = defaultdict(list) # across all datasets



for nthreads in args.nthreads:
    assert nthreads != None
    
    if args.batch_size is None:
        batch_size = nthreads
    else:
        batch_size = args.batch_size

    faiss.omp_set_num_threads(nthreads)
    print(f"Run with : {nthreads=} and {batch_size=}...")
    
    for index_name in index_name_list:

        #### Run IVF baseline
        if index_name == "IVFFlat":
            print()
            print(f"--- {index_name} ---")
            # TODO: dataset add 1 row per nlist,niter,nprobe combination, averaged across runs
            for nlist in nlist_list:
                for n_iter in n_iter_list:
                    ### BUILD INDEX
                    print()

                    # Build index for nlist niter combination
                    index, train_time, add_time = train_ivfflat(
                        xb,
                        nlist=nlist,
                        km_n_iter=n_iter,
                        seed=starting_seed,
                        verbose=True,
                        metric=metric,
                    )
                    for nprobe in nprobe_list:
                        if nprobe >= 0.75*nlist:
                            continue #skip, might as well brute force, you should have good recall with lesser nprobes at this point

                        index.nprobe = nprobe
                        latency_ms_per_batch = []
                        # lists across runs
                        qps_across_runs = []
                        recall_1_list, recall_10_list, recall_100_list = [], [], []

                        # multiple runs per nprobe to handle H/W variance
                        for run in range(nruns):
                            metric_type = "Inner Product" if index.metric_type == 0 else "L2" if index.metric_type == 1 else "Unknown"
                            print(f"{run=}, {n_iter=}, {batch_size=}, {nprobe=}, Metric: {metric_type}")

                            I = np.empty((nq, topK), dtype="int32")
                            D = np.empty((nq, topK), dtype="float32")
                            latency_per_batch = []
                            
                            i0 = 0
                            t0 = time.perf_counter()
                            while i0 < nq:
                                i1 = min(i0 + batch_size, nq)

                                t_batch_start = time.perf_counter()
                                Di, Ii = index.search(xq[i0:i1], topK)
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
                            qps = nq / total_search_time

                            qps_across_runs.append(qps)
                            recall_1_list.append(compute_recall(I, gt, 1))
                            recall_10_list.append(compute_recall(I, gt, 10))
                            recall_100_list.append(compute_recall(I, gt, 100))
                            print_recall(I, gt)


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
                            print(f"IVFFlat imbalance = {index.invlists.imbalance_factor()} & {time_entire_loop=} sec\n")

                        # append rows that is AGGREGATE across runs
                        # append row to dataframe and store .csv

                        median_qps_across_runs = np.median(qps_across_runs)
                        recall_1 = np.median(recall_1_list)
                        recall_10 = np.median(recall_10_list)
                        recall_100 = np.median(recall_100_list)
                        assert len(set(recall_1_list)) == 1, "Recall@1 values are not the same across runs"
                        assert len(set(recall_10_list)) == 1, "Recall@10 values are not the same across runs"
                        assert len(set(recall_100_list)) == 1, "Recall@100 values are not the same across runs"

                        dimensions = xq.shape[1]
                        row_to_add = {
                            "dataset_name": dbname,
                            "dim": dimensions,
                            "skew": skew,
                            "seed": starting_seed,
                            "nruns": args.nruns,
                            "index_name": index_name,
                            "nlist": nlist,
                            "nprobe": nprobe, # added later
                            "niter": n_iter,
                            "add_time": add_time,
                            "train_time": train_time,
                            "total_build_time": add_time + train_time, #train + add
                            "total_search_time": total_search_time, #sum(search)
                            "total_time": add_time + train_time + total_search_time, #train+add+sum(search)
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
                            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
                            "cmd": command,
                        }
                        # Fill any missing values with NaN
                        new_row_df = pd.DataFrame([row_to_add])
                        df = df.dropna(axis=1, how='all')
                        df = pd.concat([df, new_row_df], ignore_index=True)

                        if args.store:
                            print(f"adding row to {baseline_results_path}\n")
                            new_row_df.to_csv(baseline_results_path, mode='a', index=False, header=not pd.io.common.file_exists(baseline_results_path))

        elif index_name == "BruteForce":
            ### BUILD INDEX
            print()
            print(f"--- {index_name} ---")

            nlist = n_iter = nprobe = np.NaN
            
            index = faiss.index_factory(d, "Flat")
            start_add = time.perf_counter()
            index.add(xb)
            add_time =  (time.perf_counter() - start_add)
            train_time = 0
            
        
            latency_ms_per_batch = []
            # lists across runs
            qps_across_runs = []
            recall_1_list, recall_10_list, recall_100_list = [], [], []

            # multiple runs per nprobe to handle H/W variance
            for run in range(nruns):
                print(f"{run=}, {n_iter=}, {batch_size=}, {nprobe=}")

                I = np.empty((nq, topK), dtype="int32")
                D = np.empty((nq, topK), dtype="float32")
                latency_per_batch = []
                
                i0 = 0
                t0 = time.perf_counter()
                while i0 < nq:
                    i1 = min(i0 + batch_size, nq)

                    t_batch_start = time.perf_counter()
                    Di, Ii = index.search(xq[i0:i1], topK)
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
                qps = nq / total_search_time

                qps_across_runs.append(qps)
                recall_1_list.append(compute_recall(I, gt, 1))
                recall_10_list.append(compute_recall(I, gt, 10))
                recall_100_list.append(compute_recall(I, gt, 100))
                print_recall(I, gt)


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
                print(f"{time_entire_loop=} sec\n")

            # append rows that is AGGREGATE across runs
            # append row to dataframe and store .csv

            median_qps_across_runs = np.median(qps_across_runs)
            recall_1 = np.median(recall_1_list)
            recall_10 = np.median(recall_10_list)
            recall_100 = np.median(recall_100_list)
            assert len(set(recall_1_list)) == 1, "Recall@1 values are not the same across runs"
            assert len(set(recall_10_list)) == 1, "Recall@10 values are not the same across runs"
            assert len(set(recall_100_list)) == 1, "Recall@100 values are not the same across runs"


            dimensions = xq.shape[1]
            row_to_add = {
                "dataset_name": dbname,
                "dim": dimensions,
                "skew": skew,
                "seed": starting_seed,
                "nruns": args.nruns,
                "index_name": index_name,
                "nlist": nlist,
                "nprobe": nprobe, # added later
                "niter": n_iter,
                "add_time": add_time,
                "train_time": train_time,
                "total_build_time": add_time + train_time, #train + add
                "total_search_time": total_search_time, #sum(search)
                "total_time": add_time + train_time + total_search_time, #train+add+sum(search)
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
                "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
                "cmd": command,
            }
            # Fill any missing values with NaN
            new_row_df = pd.DataFrame([row_to_add])
            df = df.dropna(axis=1, how='all')
            df = pd.concat([df, new_row_df], ignore_index=True)

            if args.store:
                print(f"adding row to {baseline_results_path}\n")
                new_row_df.to_csv(baseline_results_path, mode='a', index=False, header=not pd.io.common.file_exists(baseline_results_path))
        
        ####NOTE: IVF Brute Force is nlist = 1, niter=0, nprobe = 1, so every query scans the same inverted list fully 
        elif index_name == "IVFBruteForce":
            print()
            print(f"--- {index_name} ---")

            for n_iter in n_iter_list:
                nlist = 1
                nprobe = 1
                niter = 0
                ### BUILD INDEX
                print()
                # Build index for nlist=1 niter=0
                index, train_time, add_time = train_ivfflat(
                    xb,
                    nlist=nlist,
                    km_n_iter=niter,
                    seed=starting_seed,
                    verbose=True,
                    metric=metric,
                )
                ## RUN SEARCH
                index.nprobe = nprobe
                latency_ms_per_batch = []
                # lists across runs
                qps_across_runs = []
                recall_1_list, recall_10_list, recall_100_list = [], [], []

                # multiple runs per nprobe to handle H/W variance
                for run in range(nruns):
                    print(f"{run=}, {n_iter=}, {batch_size=}, {nprobe=}")

                    I = np.empty((nq, topK), dtype="int32")
                    D = np.empty((nq, topK), dtype="float32")
                    latency_per_batch = []
                    
                    i0 = 0
                    t0 = time.perf_counter()
                    while i0 < nq:
                        i1 = min(i0 + batch_size, nq)

                        t_batch_start = time.perf_counter()
                        Di, Ii = index.search(xq[i0:i1], topK)
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
                    qps = nq / total_search_time

                    qps_across_runs.append(qps)
                    recall_1_list.append(compute_recall(I, gt, 1))
                    recall_10_list.append(compute_recall(I, gt, 10))
                    recall_100_list.append(compute_recall(I, gt, 100))
                    print_recall(I, gt)


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
                    print(f"IVFFlat imbalance = {index.invlists.imbalance_factor()} & {time_entire_loop=} sec\n")

                # append rows that is AGGREGATE across runs
                # append row to dataframe and store .csv

                median_qps_across_runs = np.median(qps_across_runs)
                recall_1 = np.median(recall_1_list)
                recall_10 = np.median(recall_10_list)
                recall_100 = np.median(recall_100_list)
                assert len(set(recall_1_list)) == 1, "Recall@1 values are not the same across runs"
                assert len(set(recall_10_list)) == 1, "Recall@10 values are not the same across runs"
                assert len(set(recall_100_list)) == 1, "Recall@100 values are not the same across runs"


                dimensions = xq.shape[1]
                row_to_add = {
                    "dataset_name": dbname,
                    "dim": dimensions,
                    "skew": skew,
                    "seed": starting_seed,
                    "nruns": args.nruns,
                    "index_name": index_name,
                    "nlist": nlist,
                    "nprobe": nprobe, # added later
                    "niter": n_iter,
                    "add_time": add_time,
                    "train_time": train_time,
                    "total_build_time": add_time + train_time, #train + add
                    "total_search_time": total_search_time, #sum(search)
                    "total_time": add_time + train_time + total_search_time, #train+add+sum(search)
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
                    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
                    "cmd": command,

                }
                # Fill any missing values with NaN
                new_row_df = pd.DataFrame([row_to_add])
                df = df.dropna(axis=1, how='all')
                df = pd.concat([df, new_row_df], ignore_index=True)

                if args.store:
                    print(f"adding row to {baseline_results_path}\n")
                    new_row_df.to_csv(baseline_results_path, mode='a', index=False, header=not pd.io.common.file_exists(baseline_results_path))
