"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to efficient integer matrix
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def _cut_size(labels):
        # Count edges crossing the partition
        # An edge (i,j) crosses if labels[i] != labels[j]
        # For each vertex i, count neighbors with different label
        # Sum over all vertices, divide by 2 (each edge counted twice)
        cross = np.zeros(n, dtype=np.int64)
        for v in range(n):
            cross[v] = adj[v, labels != labels[v]].sum()
        return int(cross.sum() // 2)

    def _spectral_init():
        # Use Fiedler vector for initialization
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector with second smallest eigenvalue
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            # Partition based on sign of Fiedler vector
            labels = (fiedler >= 0).astype(np.int8)
            # Ensure both partitions are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
            return labels
        except:
            return np.arange(n) % 2

    def _local_search_1flip(labels):
        # Greedy 1-flip local search
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                same = adj[v, labels == labels[v]].sum()
                other = adj[v, labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels

    def _local_search_2flip(labels):
        # 2-flip (pairwise swap) local search
        # Try flipping pairs of vertices
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        continue
                    # Compute gain of flipping both i and j
                    # Current contribution of i and j to cut
                    # After flip, i and j swap sides
                    
                    # Gain calculation:
                    # For vertex i: currently in group A, neighbors in B contribute to cut
                    # After flip to B: neighbors in A contribute to cut (excluding j if j was in A, but j is in B)
                    
                    # Let's compute the change in cut size
                    # Cut size = sum over edges (u,v) of [labels[u] != labels[v]]
                    
                    # Edges incident to i:
                    # Currently: adj[i, labels != labels[i]] contributes
                    # After flip: adj[i, labels == labels[i]] contributes (but need to adjust for j)
                    
                    # More carefully:
                    # Let A = labels[i], B = 1 - A
                    # i is in A, j is in B
                    # After flip: i in B, j in A
                    
                    # Change in cut for edges incident to i:
                    # Before: edges from i to B contribute (labels[i]=A != labels[v]=B)
                    # After: edges from i to A contribute (labels[i]=B != labels[v]=A)
                    # But edge (i,j): before i in A, j in B -> contributes 1
                    # After: i in B, j in A -> contributes 1 (no change)
                    
                    # So for i: change = adj[i, A] - adj[i, B] + 1 (for edge ij which was 1 and stays 1, so net 0 for that edge)
                    # Wait, let me think more carefully.
                    
                    # Cut contribution of edge (u,v) is 1 if labels[u] != labels[v], else 0.
                    
                    # For vertex i (currently in A):
                    # Edges (i,v) where v in B: contribute 1 each. Count = adj[i, B]
                    # Edges (i,v) where v in A: contribute 0 each.
                    # After flip, i in B:
                    # Edges (i,v) where v in A: contribute 1 each. Count = adj[i, A]
                    # Edges (i,v) where v in B: contribute 0 each.
                    # But we also flip j. j was in B, now in A.
                    
                    # This is getting complex. Let's just compute the cut before and after.
                    if time.time() > deadline:
                        break
                    
                    old_cut = _cut_size(labels)
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    new_cut = _cut_size(labels)
                    if new_cut > old_cut:
                        improved = True
                    else:
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                if time.time() > deadline:
                    break
        return labels

    # Initialize with spectral method
    labels = _spectral_init()
    
    # Run 1-flip local search
    labels = _local_search_1flip(labels)
    
    # Run 2-flip local search
    labels = _local_search_2flip(labels)
    
    # Run 1-flip local search again (polish)
    labels = _local_search_1flip(labels)
    
    # If time remains, try a few random restarts
    best_labels = labels.copy()
    best_cut = _cut_size(labels)
    
    while time.time() < deadline - 5.0:
        # Random initialization
        random_labels = np.random.randint(0, 2, size=n).astype(np.int8)
        # Ensure both partitions non-empty
        if random_labels.sum() == 0 or random_labels.sum() == n:
            random_labels = np.arange(n) % 2
        
        # Local search
        random_labels = _local_search_1flip(random_labels)
        random_labels = _local_search_2flip(random_labels)
        random_labels = _local_search_1flip(random_labels)
        
        cut_val = _cut_size(random_labels)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = random_labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
