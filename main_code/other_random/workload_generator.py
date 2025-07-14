import os
import uuid
import hashlib
import numpy as np
import pandas as pd
import faiss
import torch
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from vasili_helpers import *


class VectorWorkloadGenerator:
    def __init__(self, 
                 xb: np.ndarray, 
                 dataset_name: str, 
                 initial_size: Optional[int] = None, 
                 global_csf: float = 0.1, 
                 n_clusters: int = 10,
                 cluster_sample_ratio: Optional[float] = None,
                 seed: Optional[int] = None,
                 output_dir: str = 'workload_output'):
        """
        Initialize the Vector Workload Generator.
        """
        # Set random seed for reproducibility
        self.seed = seed if seed is not None else np.random.randint(1, 10000)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Validate cluster_sample_ratio
        if cluster_sample_ratio is not None and not (0 <= cluster_sample_ratio <= 100):
            raise ValueError("cluster_sample_ratio must be between 0 and 100")
        self.cluster_sample_ratio = cluster_sample_ratio

        # Dataset handling
        self.dataset_name = dataset_name
        self.xb = xb if initial_size is None else xb[:initial_size]
        self.initial_size = len(self.xb)

        # Generate unique identifiers
        self.run_uid = str(uuid.uuid4())
        dataset_id = (self.dataset_name if self.dataset_name != "syn"
                      else f"syn-{hashlib.md5(self.xb.tobytes()).hexdigest()[:8]}")

        # Prepare output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cluster_sample_suffix = f"clusterSample_{cluster_sample_ratio:.2f}" if cluster_sample_ratio is not None else ""
        self.output_dir = os.path.join(output_dir, 
                                       f"dataset_{dataset_id}-"
                                       f"clusters_{n_clusters}-"
                                       f"csf_{global_csf:.4f}-"
                                       f"{cluster_sample_suffix}"
                                    #    f"{timestamp}-"
                                    #    f"{self.run_uid}"
                                    )
        os.makedirs(self.output_dir, exist_ok=True)

        # Save dataset to output directory
        dataset_path = os.path.join(self.output_dir, f"xb-{dataset_id}.npy")
        np.save(dataset_path, self.xb)
        self.logger.info(f"Dataset saved at: {dataset_path}")

        self.global_csf = global_csf
        self.n_clusters = n_clusters

        # Device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Save metadata
        self._save_run_metadata()

        # Clustering and indexing attributes
        self.cluster_labels = None
        self.kmeans_centroids = None
        self.index = None

    def _generate_filename(self, base_name: str) -> str:
        """
        Generate a unique filename with run parameters
        
        Args:
            base_name (str): Base name for the file
        
        Returns:
            str: Full path to the file with unique identifier
        """
        return os.path.join(
            self.output_dir, 
            f"{base_name}-"
            f"clusters_{self.n_clusters}-"
            f"csf_{self.global_csf:.4f}-"
            # f"{self.run_uid}.npy"
        )

    def _save_run_metadata(self):
        """
        Save metadata file for the run.
        """
        metadata = {
            'run_uid': self.run_uid,
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'device': str(self.device),
            'dataset_name': self.dataset_name,
            'dataset_size': len(self.xb),
            'initial_size': self.initial_size,
            'global_csf': self.global_csf,
            'n_clusters': self.n_clusters,
            'cluster_sample_ratio': self.cluster_sample_ratio
        }
        with open(os.path.join(self.output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        self.logger.info(f"Run metadata saved. Unique ID: {self.run_uid}")

    def cluster_dataset(self, niter: int = 25, nredo: int = 1) -> Dict:
        """
        Perform k-means clustering on the dataset
        
        Args:
            niter (int): Number of iterations for k-means
        
        Returns:
            Dict with clustering information
        """
        d = self.xb.shape[1]  # dimension of vectors
        
        # Use FAISS for GPU/CPU k-means clustering
        clust = faiss.Kmeans(d, self.n_clusters, niter=niter, seed=self.seed, nredo=nredo, verbose=True)
        clust.train(self.xb.astype('float32'))

        # Compute cluster assignments
        _, self.cluster_labels = clust.assign(self.xb.astype('float32'))
        self.kmeans_centroids = clust.centroids
        
        # Prepare cluster mapping
        cluster_mapping = {}
        for i, label in enumerate(self.cluster_labels):
            if label not in cluster_mapping:
                cluster_mapping[label] = []
            cluster_mapping[label].append(i)
        
        # Log clustering results
        self.logger.info(f"Clustered dataset into {self.n_clusters} clusters")
        
        # Store cluster statistics
        cluster_stats = {
            'centroids': self.kmeans_centroids,
            'labels': self.cluster_labels,
            'mapping': cluster_mapping
        }
        
        # Save clustering results with unique naming
        np.save(self._generate_filename('cluster_labels'), self.cluster_labels)
        np.save(self._generate_filename('cluster_centroids'), self.kmeans_centroids)
        
        return cluster_stats

    def generate_queries(self, 
                        n_queries: int = 100, 
                        k: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate queries with ground truth k-NN ensuring exact number of queries.
        """
        # Ensure clustering has been performed
        if self.cluster_labels is None:
            self.cluster_dataset()
        
        # Create FAISS index for ground truth computation
        d = self.xb.shape[1]
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(self.xb.astype('float32'))
            index = gpu_index
        else:
            index = faiss.IndexFlatL2(d)
            index.add(self.xb.astype('float32'))
        
        query_indices = set()
        query_cluster_ids = []

        # NOTE: keep sampling until we reach target n_queries
        #   - this is needed if global_csf is not correct and you have cluster sample ratios etc.
        #   - slight bias for first clusters to have higher skew (as when you loop you start sampling again from the first ones)
        while len(query_indices) < n_queries:
        # for i in range(1):
            # Reset indices for this iteration
            iteration_indices = []
            iteration_cluster_ids = []

            # Determine unique clusters
            unique_clusters = np.unique(self.cluster_labels)

            # Apply cluster sampling ratio if specified
            if self.cluster_sample_ratio is not None:
                # Calculate number of clusters to sample
                n_clusters_to_sample = max(1, int(len(unique_clusters) * (self.cluster_sample_ratio / 100)))
                sampled_clusters = np.random.choice(
                    unique_clusters, 
                    size=n_clusters_to_sample, 
                    replace=False
                )
            else:
                sampled_clusters = unique_clusters

            # Sample queries from selected clusters
            for cluster in sampled_clusters:
                cluster_indices = np.where(self.cluster_labels == cluster)[0]
                
                # Determine number of queries from this cluster
                n_cluster_queries = max(1, int(n_queries * (len(cluster_indices) / len(self.xb))))
                
                # Sample queries from the cluster
                cluster_query_indices = np.random.choice(
                    cluster_indices, 
                    size=min(n_cluster_queries, len(cluster_indices)), 
                    replace=False
                )
                
                iteration_indices.extend(cluster_query_indices)
                iteration_cluster_ids.extend([cluster] * len(cluster_query_indices))

            # Add new unique indices
            query_indices.update(iteration_indices)
            query_cluster_ids.extend(iteration_cluster_ids[:len(iteration_indices)])

        # Finalize queries ensuring the count matches
        query_indices = list(query_indices)[:n_queries]
        xq = self.xb[query_indices]
        query_cluster_ids = query_cluster_ids[:n_queries]
        
        # Compute ground truth
        print("Computing Ground Truth...")
        D, I = index.search(xq.astype('float32'), k)
        
        # Prepare metadata DataFrame
        query_metadata = pd.DataFrame({
            'query_index': query_indices,
            'query_cluster': query_cluster_ids,
        })
        
        # Save metadata
        query_metadata_filename = os.path.join(
            self.output_dir, 
            f"query_metadata-"
            f"nqueries_{n_queries}-"
            f"knn_{k}-"
            f"clusterSample_{self.cluster_sample_ratio}-"
            # f"{self.run_uid}.csv"
        )
        query_metadata.to_csv(query_metadata_filename, index=False)
        np.save(self._generate_filename('xq'), xq)
        np.save(self._generate_filename('gt'), I)
        
        return {
            'queries': xq,
            'ground_truth_indices': I,
            'metadata': query_metadata
        }


    def generate_workload(self, 
                           n_queries: int = 100, 
                           k: int = 10) -> Dict:
        """
        Generate complete vector search workload
        """
        # Cluster the dataset first
        cluster_stats = self.cluster_dataset()
        
        # Generate queries with ground truth
        query_data = self.generate_queries(n_queries, k)
        
        # Prepare comprehensive workload metadata
        workload_metadata = {
            'cluster_stats': cluster_stats,
            'queries': query_data,
        }

        return workload_metadata


def parse_arguments():
    """
    Parse command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Vector Workload Generator")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name (available: [syn, sift1m]) (Use 'syn' for synthetic datasets) .")
    parser.add_argument("--output_dir", type=str, default="workload_output", 
                        help="Output directory for generated files.")
    parser.add_argument("--global_csf", type=float, default=0.1, 
                        help="Global cluster sampling factor.")
    parser.add_argument("--n_clusters", type=int, default=10, 
                        help="Number of clusters.")
    parser.add_argument("--cluster_sample_ratio", type=float, default=None, 
                        help="Cluster sampling ratio (percentage).")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed.")
    parser.add_argument("--initial_size", type=int, default=None, 
                        help="Initial dataset size to use.")
    parser.add_argument("--num_queries", type=int, default=100, 
                        help="Number of queries to generate (default 100)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load or create the dataset
    if args.dataset.lower() == "syn":
        d = 128  # dimension
        nb = 10000  # database size
        xb = np.random.random((nb, d)).astype('float32')
        dataset_name = "syn"
    elif args.dataset.lower() == "sift1m":
        xb, _, _, _, _ = load_sift1M(f"/pub/scratch/vmageirakos/datasets/ann-fvecs/sift-128-euclidean")
        dataset_name="sift1m"

    generator = VectorWorkloadGenerator(
        xb=xb,
        dataset_name=dataset_name,
        initial_size=args.initial_size,
        global_csf=args.global_csf,
        n_clusters=args.n_clusters,
        cluster_sample_ratio=args.cluster_sample_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )

    generator.generate_workload(n_queries=args.num_queries, k=100)
    print("Done.")

if __name__ == '__main__':
    main()