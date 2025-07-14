# crack-ivf-vldb

# setup enviroment

We extended faiss with a few functions to add mutability on IVFFlat

You must compile the faiss, under `faiss-crack`:
- our exact conda/mamba env used is under `/faiss-crack/crackivf.yml`
- activate env
- run `install_faiss_mkl_release.sh --python` to compile w/ swig bindings

# run code

under `/main_code`:
- `./run_ours.sh` : runs our measurements
    - results stored under `/results/<dataset>_default_OURS/...`
- `./run_baselines.sh` : runs baseline measurements
    - results stored under `/results/<dataset>_default_BASELINE/...`

- Otherwise if you want custom runs, change this command with arguments you'd like:
    - `python run_ours.py --nthreads 16 --store --detailed --clear_results --target_queries 100000 --nruns 1 --nlist 100 --niter 10 --nprobe 99999 --get_qps  --dbname SIFT1M --run_desc your_run`


# get plots
- `python plot_all_for_paper.py --skew default --plotid some_uniq_id --yscale log  --dbname SIFT1M` 
    - change the SIFT1M to whichever dataset you'd like

# paper results
some of the plots and measurements appearing in the paper can be found under `/results`. The more recent ones are under https://polybox.ethz.ch/index.php/s/wa7nJG2YQroAWFD 

# dataset sources
sift1b, which we slice, to 1M/10M   
SITF (bigann) dataset: http://corpus-texmex.irisa.fr/  
```
mkdir bigann
cd bigann
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz 
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz
gunzip bigann_learn.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip bigann_query.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar xzvf bigann_gnd.tar.gz
rm bigann_gnd.tar.gz
```
  
ann-benchmarks click and downlod the HDF5 : https://github.com/erikbern/ann-benchmarks  
glove (25/50/100), deep, last.fm   
