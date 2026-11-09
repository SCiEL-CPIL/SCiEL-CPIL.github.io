"""Maximum Cut via SDP-inspired spectral embedding + randomized hyperplane + fast local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    if n == 0:
        return np.array([], dtype=np.int8)

    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def flip_greedy(labels, max_passes=50):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        # Precompute for each vertex: count of neighbors in same partition
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                # Gain of flipping v: 
                # Before flip: v has (deg - same_count[v]) crossing edges
                # After flip: v has same_count[v] crossing edges
                # Gain = same_count[v] - (deg[v] - same_count[v]) = 2*same_count[v] - deg[v]
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    # Update same_count for v and its neighbors
                    # v's new same_count = deg - old_same_count
                    new_same_v = deg - same_count[v]
                    same_count[v] = new_same_v
                    # For each neighbor u of v:
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            # After flip, u and v are same, before they were different
                            same_count[u] += 1
                        else:
                            # After flip, u and v are different, before they were same
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def two_swap_search(labels, max_passes=20):
        """2-swap local search with efficient gain computation."""
        labels = labels.copy()
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    if labels[i] == labels[j]:
                        continue
                    
                    li = labels[i]
                    lj = labels[j]
                    
                    # n_ii = neighbors of i with label li
                    # n_ij = neighbors of i with label lj
                    # n_ji = neighbors of j with label li
                    # n_jj = neighbors of j with label lj
                    
                    n_ii = 0
                    n_ij = 0
                    for u in neighbors[i]:
                        if labels[u] == li:
                            n_ii += 1
                        else:
                            n_ij += 1
                    
                    n_ji = 0
                    n_jj = 0
                    for u in neighbors[j]:
                        if labels[u] == li:
                            n_ji += 1
                        else:
                            n_jj += 1
                    
                    # Edge (i,j) is in n_ij and n_ji if adj[i,j]=1
                    e_ij = adj[i, j]
                    
                    # Gain = n_ii + n_jj - n_ij - n_ji + 2*e_ij
                    gain = n_ii + n_jj - n_ij - n_ji + 2 * e_ij
                    
                    if gain > 0:
                        labels[i] = lj
                        labels[j] = li
                        improved = True
            
            if not improved:
                break
        
        return labels

    # Compute spectral embedding for Goemans-Williamson style initialization
    # Use the top k eigenvectors to create a low-dimensional embedding
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        # Sort by eigenvalue (descending) - eigh returns ascending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Use top k eigenvectors for embedding
        k = min(10, n)
        embedding = eigenvectors[:, :k]
        
        # Normalize rows to unit length (like SDP solution)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embedding = embedding / norms
    except Exception:
        embedding = None

    best_cut = -1
    best_labels = None

    # Generate multiple initial partitions using randomized hyperplane projection
    # on the spectral embedding, plus fallbacks
    init_labels_list = []
    
    # 1. Principal eigenvector sign (classic spectral)
    if embedding is not None:
        v0 = embedding[:, 0]
        init_labels_list.append((v0 > 0).astype(np.int8))
        init_labels_list.append((v0 < 0).astype(np.int8))
        
        # 2. Randomized hyperplane projections on the embedding
        num_hyperplanes = 15
        for h in range(num_hyperplanes):
            if time.time() > deadline:
                break
            # Random unit vector in R^k
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            init_labels_list.append((projection > 0).astype(np.int8))
    
    # 3. Fallback: alternating pattern
    init_labels_list.append(np.arange(n) % 2)
    
    # 4. Fallback: random
    init_labels_list.append(np.random.randint(0, 2, size=n, dtype=np.int8))

    # Process each initial label assignment
    for init_labels in init_labels_list:
        if time.time() > deadline:
            break
        
        # 1-flip greedy
        labels1 = flip_greedy(init_labels, max_passes=30)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()
        
        # 2-swap after 1-flip
        if time.time() < deadline:
            labels2 = two_swap_search(labels1, max_passes=8)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()

    # Additional random restarts with pure random initialization
    num_random = 20
    for r in range(num_random):
        if time.time() > deadline:
            break
        
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        
        # 1-flip greedy
        labels = flip_greedy(labels, max_passes=25)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # 2-swap
        if time.time() < deadline:
            labels_sw = two_swap_search(labels, max_passes=6)
            cut_sw = compute_cut(labels_sw)
            if cut_sw > best_cut:
                best_cut = cut_sw
                best_labels = labels_sw.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
