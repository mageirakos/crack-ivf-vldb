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

### RUN 2 ### RUN 2 ##### RUN 2 ##### RUN 2 ##### RUN 2 ##### RUN 2 ##### RUN 2 ##### RUN 2 ##### RUN 2

# SIFT1M
run_command "python run_ours.py --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"

run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"


# # glove-25-angular
run_command "python run_ours.py --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run0-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"


# glove-50-angular
run_command "python run_ours.py --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run0-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"

# glove-100-angular
run_command "python run_ours.py  --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run0-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"

# lastfm-64-dot
run_command "python run_ours.py --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run0-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"

# deep-image-96-angular (eucl. dist)
run_command "python run_ours.py --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run0-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"

# SIFT10M
run_command "python run_ours.py --run_desc 20250531_final_abl_run0 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run0-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run0-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run0-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run0-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run0-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run0-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run0-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run0-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run0-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"







##### RUN 3 ##### RUN 3 ##### RUN 3 ##### RUN 3 ##### RUN 3 ##### RUN 3 ##### RUN 3 ##### RUN 3

# SIFT1M
run_command "python run_ours.py --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT1M --nthreads 16 --nlist 100"


# # glove-25-angular
run_command "python run_ours.py --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-25-angular --nthreads 16 --nlist 100"


# glove-50-angular
run_command "python run_ours.py --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-50-angular --nthreads 16 --nlist 100"

# glove-100-angular
run_command "python run_ours.py  --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname glove-100-angular --nthreads 16 --nlist 100"

# lastfm-64-dot
run_command "python run_ours.py --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname lastfm-64-dot --nthreads 16 --nlist 100"

# deep-image-96-angular (eucl. dist)
run_command "python run_ours.py --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname deep-image-96-angular --nthreads 16 --nlist 100"

# SIFT10M
run_command "python run_ours.py --run_desc 20250531_final_abl_run1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 1  --run_desc 20250531_final_abl_run1-min_pts_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 16  --run_desc 20250531_final_abl_run1-min_pts_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --min_pts 32  --run_desc 20250531_final_abl_run1-min_pts_32 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 16  --run_desc 20250531_final_abl_run1-pts_crack_thr_16 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --pts_crack_thr 128  --run_desc 20250531_final_abl_run1-pts_crack_thr_128 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 1 --run_desc 20250531_final_abl_run1-cv_max_1 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --cv_max 8 --run_desc 20250531_final_abl_run1-cv_max_8 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 1 --size_prcnt_high 99 --run_desc 20250531_final_abl_run1-size_prcnt_1_99 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
run_command "python run_ours.py --size_prcnt_low 25 --size_prcnt_high 75 --run_desc 20250531_final_abl_run1-size_prcnt_25_75 --get_qps --store --detailed --clear_results --target_queries 200000 --nruns 1 --niter 10 --nprobe 99999  --dbname SIFT10M --nthreads 16 --nlist 100"
