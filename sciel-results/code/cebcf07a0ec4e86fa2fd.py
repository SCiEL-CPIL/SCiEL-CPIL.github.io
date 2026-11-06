"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    # Convert to numpy int8 for fast operations
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: return best result before 145s wall-clock
    start_time = time.time()
    deadline = start_time + 145.0
    
    def cut_size(labels):
        # Count edges crossing the partition
        # For each edge (i,j), it crosses if labels[i] != labels[j]
        # Using vectorized computation
        return int(np.sum(adj[labels == 0, :].sum(axis=1)) - 0)  # placeholder, will use better method
    
    def actual_cut_size(labels):
        # More efficient: for each vertex in group 0, count neighbors in group 1
        group0 = (labels == 0)
        group1 = (labels == 1)
        return int(np.sum(adj[group0, :].sum(axis=1)))
    
    def flip_gain(v, labels):
        """Gain from flipping vertex v. Positive gain means flip improves cut."""
        # Current contribution of v to cut: edges from v to opposite group
        # After flip: edges from v to same group (which becomes opposite)
        # Gain = (edges to same group) - (edges to opposite group)
        same_group = (labels == labels[v]).sum()
        # Actually: gain = 2*(edges to opposite) - degree
        # Because: new_cut - old_cut = edges_to_same - edges_to_opposite
        # edges_to_opposite = degree - edges_to_same
        # gain = edges_to_same - (degree - edges_to_same) = 2*edges_to_same - degree
        # Wait, let me reconsider.
        # If v is in group 0:
        #   Current: edges from v to group 1 contribute to cut
        #   After flip to group 1: edges from v to group 0 contribute to cut
        #   Gain = (edges to group 0) - (edges to group 1)
        # edges to group 0 = same_group_count (excluding self, but adj[v,v]=0)
        # edges to group 1 = degree - same_group_count
        # Gain = same - (deg - same) = 2*same - deg
        deg = adj[v].sum()
        same = adj[v][labels == labels[v]].sum()
        return 2 * same - deg
    
    def local_search_1flip(labels, max_iters=100):
        """Greedy 1-flip local search."""
        labels = labels.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if time.time() > deadline:
                    break
                gain = flip_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
            if time.time() > deadline:
                break
        return labels
    
    def local_search_2flip(labels, max_iters=50):
        """Local search allowing 2-flip moves."""
        labels = labels.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_gain = 0
            best_pair = None
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    # Compute gain from flipping both i and j
                    # This is more complex; approximate by checking individual gains first
                    gain_i = flip_gain(i, labels)
                    gain_j = flip_gain(j, labels)
                    # For 2-flip, the interaction term matters
                    # Gain from flipping i and j together:
                    # = gain_i + gain_j + 2*adj[i,j] (if they are adjacent, the edge between them changes contribution)
                    # Actually, need to be careful.
                    # Let's just try the pair if either has positive gain
                    if gain_i > 0 or gain_j > 0:
                        # Tentatively flip both
                        old_li = labels[i]
                        old_lj = labels[j]
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                        # Compute new cut size difference
                        # This is expensive, so let's use a heuristic
                        labels[i] = old_li
                        labels[j] = old_lj
                        # For simplicity, only do 1-flip in this version
                        pass
            # Fall back to 1-flip if no 2-flip found
            for v in range(n):
                if time.time() > deadline:
                    break
                gain = flip_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
            if time.time() > deadline:
                break
        return labels
    
    def spectral_init():
        """Initialize using Fiedler vector of the Laplacian."""
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is the second smallest eigenvector
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            # Partition based on sign of Fiedler vector
            labels = (fiedler >= 0).astype(np.int8)
            # Ensure both groups are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
            return labels
        except:
            return np.arange(n) % 2
    
    def random_init():
        """Random initialization."""
        return np.random.randint(0, 2, size=n).astype(np.int8)
    
    def alternating_init():
        """Alternating initialization."""
        return np.arange(n) % 2
    
    # Generate multiple initial partitions and refine each
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral init
    labels1 = spectral_init()
    labels1 = local_search_1flip(labels1)
    cut1 = actual_cut_size(labels1)
    if cut1 > best_cut:
        best_cut = cut1
        best_labels = labels1.copy()
    
    # Strategy 2: Alternating init
    labels2 = alternating_init()
    labels2 = local_search_1flip(labels2)
    cut2 = actual_cut_size(labels2)
    if cut2 > best_cut:
        best_cut = cut2
        best_labels = labels2.copy()
    
    # Strategy 3: Random inits
    num_random = 5
    for r in range(num_random):
        if time.time() > deadline:
            break
        labels_r = random_init()
        labels_r = local_search_1flip(labels_r)
        cut_r = actual_cut_size(labels_r)
        if cut_r > best_cut:
            best_cut = cut_r
            best_labels = labels_r.copy()
    
    # Interleaved 1-flip and 2-flip refinement on best solution
    if time.time() < deadline:
        # Do more rounds of local search with occasional 2-flip attempts
        for round_num in range(5):
            if time.time() > deadline:
                break
            improved = True
            while improved:
                improved = False
                # 1-flip pass
                for v in range(n):
                    if time.time() > deadline:
                        break
                    gain = flip_gain(v, best_labels)
                    if gain > 0:
                        best_labels[v] = 1 - best_labels[v]
                        improved = True
                if time.time() > deadline:
                    break
                # Try some 2-flip moves
                num_2flip_attempts = min(100, n * n // 10)
                for _ in range(num_2flip_attempts):
                    if time.time() > deadline:
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    # Compute gain from flipping both i and j
                    # Old contribution: edges from i to opp(i) + edges from j to opp(j)
                    # New contribution: edges from i to same(i) + edges from j to same(j)
                    # Plus interaction: edge (i,j) changes status
                    old_cut = actual_cut_size(best_labels)
                    best_labels[i] = 1 - best_labels[i]
                    best_labels[j] = 1 - best_labels[j]
                    new_cut = actual_cut_size(best_labels)
                    if new_cut > old_cut:
                        improved = True
                    else:
                        best_labels[i] = 1 - best_labels[i]
                        best_labels[j] = 1 - best_labels[j]
    
    # Final polish: more intensive local search
    if time.time() < deadline:
        best_labels = local_search_1flip(best_labels, max_iters=200)
    
    return best_labels
# EVOLVE-BLOCK-END
