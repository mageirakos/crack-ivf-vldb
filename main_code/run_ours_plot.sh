#!/bin/bash
# Start plots

# This script finds all ablation runs in SCRIPT
# copies the correct result files for each run, and generates the plots.

SCRIPT="plot_this_cmd.log" # NOTE: change this to what you want to plot, rundesc and dbname will be extracted

# Function to copy result files and plot for a given dbname and run_desc
copy_and_plot() {
    local dbname="$1"
    local run_desc="$2"
    local dir="results/$RESULTS_DIR/${dbname}_default_OURS"

    echo ""
    echo "=== Processing: DB = $dbname | Run = $run_desc ==="
    # fail if any .csv files are not found
    if [[ ! -d "$dir" ]]; then
        echo "Directory $dir does not exist. Skipping."
        return
    fi
    cp "$dir/qps_summary_results_${run_desc}.csv" "$dir/qps_summary_results.csv" # forgot to run QPS
    cp "$dir/summary_results_${run_desc}.csv" "$dir/summary_results.csv"
    cp "$dir/super_detailed_results_${run_desc}.csv" "$dir/super_detailed_results.csv"
    
    python plot_all_for_paper.py --skew default --plotid "$run_desc" --yscale log --dbname "$dbname" # paper plots
    # python plot_all.py --skew default --plotid "$run_desc" --yscale log --dbname "$dbname"
}

# Main loop: find all ablation runs and process them
grep -oP 'run_command "\Kpython run_ours.py[^"]+' "$SCRIPT" | while read -r cmd; do
    dbname=$(echo "$cmd" | grep -oP '(?<=--dbname )\S+')
    run_desc=$(echo "$cmd" | grep -oP '(?<=--run_desc )\S+')

    # Only process if both are found
    if [[ -n "$dbname" && -n "$run_desc" ]]; then
        copy_and_plot "$dbname" "$run_desc"
    fi
done