#!/bin/bash
# 
# To add more datasets, copy and paste the following line:
# run_command "python run_baselines.py --store --dbname <your-dbname>"

# Function to run a command and check for errors
run_command() {
    echo "Running: $1"
    eval $1

    if [ $? -ne 0 ]; then
        echo "Error: Command failed. Exiting."
        exit 1
    fi

    echo "Command completed successfully: $1"
}



# 25/02/25 - re-run final for paper
run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname SIFT1M" 
run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname SIFT10M" 
run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname glove-25-angular"
run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname glove-50-angular"
run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname glove-100-angular"
run_command "python run_ours.py --run_desc 20250227-1M-alpha_0.5-MINPTS_2-Pts2CentThres_64-conv_200-nthread_16 --nthreads 16 --store --detailed --clear_results --target_queries 1000000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname deep-image-96-angular" # euclidean actually