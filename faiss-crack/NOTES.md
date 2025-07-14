For good performance in the .add() operations when initializing the IVF, as well as during high batch search.

It is very important to have Intel MKL installed.

They way we do it here is using conda enviroments:

1) install intel mkl with conda
2) run the ./install_faiss_mkl_releas.sh bash script which should find the /lib folder of your currently activated conda enviroment, where it will find Intel MKL
3) If it doesn't find it, it falls back to traditional sgemm