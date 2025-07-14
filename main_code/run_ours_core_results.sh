#!/bin/bash
# 
# To add more datasets, copy and paste the following line:
# run_command "python run_baselines.py --get_qps --store --dbname <your-dbname>"

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


# REVISION_final == RULE_2_OFF
# Vary parameters

# RUN 1
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_run1 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname SIFT10M --nthreads 16 --nlist 100"

# RUN 2
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_run2 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run2 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run2 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run2 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run2 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run2 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname SIFT10M --nthreads 16 --nlist 100"

# RUN 3
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_run3 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run3 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run3 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run3 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run3 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --run_desc 20250601_RULE2OFF_REVISION_final_abl_run3 --get_qps --store --detailed --clear_results --target_queries 1000000 --nruns 1 --niter 10 --nprobe 99999 --dbname SIFT10M --nthreads 16 --nlist 100"
