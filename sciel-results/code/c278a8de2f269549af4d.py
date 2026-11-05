import itertools
import time
import numpy as np
import random
import heapq

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
        
    # Build reverse index: tsub -> list of blocks
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for j in range(num_kblocks):
        for idx in block_tsub_idxs[j]:
            tsub_to_blocks[idx].append(j)

    def polish(selected):
        if not selected:
            return selected
            
        counts = np.zeros(num_tsubs, dtype=np.int8)
        for b_idx in selected:
            for idx in block_tsub_idxs[b_idx]:
                counts[idx] += 1
                
        changed = True
        while changed and time.monotonic() < deadline - 5.0:
            changed = False
            new_selected = []
            for b_idx in selected:
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected = new_selected
            if not changed:
                break
                
        return selected

    def greedy_construct_lazy():
        """
        Greedy Set Cover with lazy evaluation using a priority queue.
        """
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        heap = []
        for j in range(num_kblocks):
            heapq.heappush(heap, (-len(block_tsub_idxs[j]), j))
            
        while time.monotonic() < deadline - 10.0:
            if np.all(covered):
                break
                
            best_gain = -1
            best_block = -1
            
            while heap:
                neg_gain, j = heapq.heappop(heap)
                current_gain = -neg_gain
                
                actual_gain = 0
                for idx in block_tsub_idxs[j]:
                    if not covered[idx]:
                        actual_gain += 1
                
                if actual_gain == current_gain and actual_gain > 0:
                    best_gain = actual_gain
                    best_block = j
                    break
                else:
                    if actual_gain > 0:
                        heapq.heappush(heap, (-actual_gain, j))
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            for idx in block_tsub_idxs[best_block]:
                if not covered[idx]:
                    covered[idx] = True
        
        return selected

    def build_spectral_embedding(selected_blocks):
        """
        Build a weighted graph on selected blocks where edge weight = number of shared t-subsets.
        Compute Laplacian eigenvectors to identify clusters of high-overlap blocks.
        Returns:
            - cluster_ids: array of cluster assignments for each selected block
            - embedding: spectral embedding coordinates for each selected block
        """
        n = len(selected_blocks)
        if n < 2:
            return np.zeros(n, dtype=int), np.zeros((n, 2))
        
        # Build adjacency matrix (weighted by shared t-subsets)
        # For efficiency, only compute edges between blocks that share at least one t-subset
        adj = np.zeros((n, n), dtype=np.int16)
        
        # For each selected block, find other selected blocks that share t-subsets
        block_to_pos = {b: i for i, b in enumerate(selected_blocks)}
        
        for i, b_i in enumerate(selected_blocks):
            for idx in block_tsub_idxs[b_i]:
                for b_j in tsub_to_blocks[idx]:
                    if b_j in block_to_pos:
                        j = block_to_pos[b_j]
                        if j != i:
                            adj[i, j] += 1
        
        # Symmetrize (should already be symmetric)
        adj = (adj + adj.T) // 2
        
        # Compute degree matrix
        degrees = adj.sum(axis=1)
        
        # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        # Use sparse Laplacian for efficiency
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        D_inv_sqrt = np.zeros((n, n))
        for i in range(n):
            if degrees[i] > 0:
                D_inv_sqrt[i, i] = 1.0 / np.sqrt(degrees[i])
        
        L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Get smallest eigenvalues/eigenvectors (excluding trivial ones)
        n_eigs = min(3, n - 1)  # Use 2-3 dimensions
        if n_eigs < 1:
            return np.zeros(n, dtype=int), np.zeros((n, 2))
        
        try:
            # eigsh finds smallest magnitude eigenvalues by default with which='SA'
            # We want smallest positive eigenvalues (excluding 0)
            evals, evecs = eigsh(L, k=n_eigs + 1, which='SA')
            
            # Sort by eigenvalue
            idx_sort = np.argsort(evals)
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]
            
            # Skip the first eigenvector (should be constant)
            if evals[0] < 1e-10:
                embedding = evecs[:, 1:1+n_eigs]
            else:
                embedding = evecs[:, :n_eigs]
                
        except Exception:
            # Fallback: use simple clustering based on overlap
            embedding = np.random.randn(n, 2)
        
        # Cluster using simple threshold on embedding
        # Blocks that are close in embedding space are in the same cluster
        cluster_ids = np.zeros(n, dtype=int)
        if n > 1:
            # Use simple k-means-like clustering with k=2-4
            n_clusters = min(4, n)
            # Initialize centroids randomly
            centroids = embedding[random.sample(range(n), n_clusters)]
            
            for _ in range(10):  # 10 iterations
                # Assign to nearest centroid
                dists = np.linalg.norm(embedding[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
                cluster_ids = np.argmin(dists, axis=1)
                
                # Update centroids
                for c in range(n_clusters):
                    mask = cluster_ids == c
                    if mask.any():
                        centroids[c] = embedding[mask].mean(axis=0)
        
        return cluster_ids, embedding

    def spectral_local_search(selected):
        """
        Local search that uses spectral embedding to guide perturbations.
        Blocks in the same spectral cluster are more likely to be redundant,
        so we preferentially remove blocks from high-overlap clusters.
        """
        best_selected = polish(selected[:])
        best_len = len(best_selected)
        
        current_selected = best_selected[:]
        no_improve = 0
        max_no_improve = 30
        
        while time.monotonic() < deadline - 2.0 and no_improve < max_no_improve:
            candidate = current_selected[:]
            
            if not candidate:
                break
            
            # Build spectral embedding to guide perturbation
            cluster_ids, embedding = build_spectral_embedding(candidate)
            
            # Compute cluster sizes
            unique_clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
            cluster_size = {c: cnt for c, cnt in zip(unique_clusters, cluster_counts)}
            
            # Perturb: Remove a block, preferring those in larger clusters (higher redundancy)
            # Weight removal probability by cluster size
            removal_weights = np.array([cluster_size.get(cluster_ids[i], 1) for i in range(len(candidate))], dtype=float)
            removal_weights /= removal_weights.sum()
            
            rem_idx = np.random.choice(len(candidate), p=removal_weights)
            candidate.pop(rem_idx)
            
            # Repair with greedy
            covered = np.zeros(num_tsubs, dtype=bool)
            candidate_set = set(candidate)
            for b_idx in candidate:
                for idx in block_tsub_idxs[b_idx]:
                    covered[idx] = True
            
            # Fast greedy repair: add blocks with highest marginal gain
            for _ in range(5):
                if np.all(covered):
                    break
                best_gain = -1
                best_block = -1
                
                # Sample a subset of blocks to speed up
                if num_kblocks > 10000:
                    sample_size = min(5000, num_kblocks)
                    candidates_to_check = random.sample(range(num_kblocks), sample_size)
                else:
                    candidates_to_check = range(num_kblocks)
                
                for j in candidates_to_check:
                    if j in candidate_set:
                        continue
                    gain = 0
                    for idx in block_tsub_idxs[j]:
                        if not covered[idx]:
                            gain += 1
                            if gain > best_gain:
                                if gain == 35:
                                    break
                    if gain > best_gain:
                        best_gain = gain
                        best_block = j
                        
                if best_block == -1 or best_gain == 0:
                    break
                candidate.append(best_block)
                candidate_set.add(best_block)
                for idx in block_tsub_idxs[best_block]:
                    covered[idx] = True

            candidate = polish(candidate)
            
            if len(candidate) < best_len:
                best_selected = candidate[:]
                best_len = len(candidate)
                current_selected = candidate[:]
                no_improve = 0
            else:
                no_improve += 1
                current_selected = best_selected[:]
                
        return best_selected

    def greedy_random_tsub():
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        uncovered_list = list(range(num_tsubs))
        
        while uncovered_list and time.monotonic() < deadline - 10.0:
            idx_pos = random.randint(0, len(uncovered_list) - 1)
            tsub_idx = uncovered_list[idx_pos]
            
            blocks = tsub_to_blocks[tsub_idx]
            if not blocks:
                break
            block_idx = blocks[random.randint(0, len(blocks) - 1)]
            
            selected.append(block_idx)
            
            for idx in block_tsub_idxs[block_idx]:
                covered[idx] = True
            
            new_uncovered = [u for u in uncovered_list if not covered[u]]
            uncovered_list = new_uncovered
            
        return selected

    # Main Loop
    # 1. Initial Greedy (Lazy)
    initial = greedy_construct_lazy()
    initial = polish(initial)
    
    # 2. Spectral Local Search
    best = spectral_local_search(initial)
    
    # 3. Restarts with random tsub greedy
    for i in range(5):
        if time.monotonic() > deadline - 20.0:
            break
        candidate = greedy_random_tsub()
        candidate = polish(candidate)
        candidate = spectral_local_search(candidate)
        
        if len(candidate) < len(best):
            best = candidate
            
    # Final polish
    best = polish(best)
    
    # Convert indices to lists
    result_blocks = [list(all_kblocks[i]) for i in best]
    
    return result_blocks
