"""Maximum Cut via Spectral Initialization, Fast Local Search, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral (Fiedler vector) initialization, fast local search with
    2-flip neighborhood, and simulated annealing refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees and neighbor lists for efficiency
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]

    def compute_cut(labels):
        """Compute the total cut size."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        return int(in_group0 @ adj @ in_group1)

    def compute_same(labels):
        """Compute same[v] = number of neighbors of v with same label as v."""
        same = np.zeros(n, dtype=np.int32)
        for v in range(n):
            if labels[v] == 0:
                same[v] = int(adj[v, labels == 0].sum())
            else:
                same[v] = int(adj[v, labels == 1].sum())
        return same

    def local_search(labels, time_limit):
        """
        Run local search with batched non-conflicting 1-flip moves and 2-flip moves.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        same = compute_same(labels)
        
        improved = True
        two_flip_count = 0
        max_two_flips = 200
        stall_count = 0
        max_stalls = 50
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Compute gains for all vertices
            gains = 2 * same - degrees
            
            # Find all vertices with positive gain
            positive = np.where(gains > 0)[0]
            
            if len(positive) > 0:
                # Batch non-conflicting flips: greedily select vertices whose neighbors don't conflict
                flipped = set()
                to_flip = []
                for v in positive:
                    if v in flipped:
                        continue
                    # Check if any neighbor is already scheduled to flip
                    conflict = False
                    for u in neighbors[v]:
                        if u in flipped:
                            conflict = True
                            break
                    if not conflict:
                        to_flip.append(v)
                        flipped.add(v)
                
                if to_flip:
                    # Apply all flips in batch
                    for v in to_flip:
                        old_label = labels[v]
                        labels[v] = 1 - old_label
                        new_label = labels[v]
                        
                        # Update same[v]
                        if new_label == 0:
                            same[v] = int(adj[v, labels == 0].sum())
                        else:
                            same[v] = int(adj[v, labels == 1].sum())
                    
                    # Update neighbors for all flipped vertices
                    for v in to_flip:
                        old_label = 1 - labels[v]  # original label before flip
                        new_label = labels[v]
                        for u in neighbors[v]:
                            if u == v or u in flipped:
                                continue
                            if labels[u] == old_label:
                                same[u] -= 1
                            elif labels[u] == new_label:
                                same[u] += 1
                    
                    # Recompute cut
                    cut = compute_cut(labels)
                    improved = True
                    stall_count = 0
                    continue
            
            # No 1-flip improvements; try 2-flip moves
            if two_flip_count < max_two_flips:
                two_flip_count += 1
                best_gain = 0
                best_v1, best_v2 = -1, -1
                
                # Scan all pairs for best 2-flip gain
                for u in range(n):
                    gain1_u = 2 * same[u] - degrees[u]
                    for v in range(u + 1, n):
                        gain1_v = 2 * same[v] - degrees[v]
                        if adj[u, v] == 1:
                            if labels[u] == labels[v]:
                                gain2 = gain1_u + gain1_v - 2
                            else:
                                gain2 = gain1_u + gain1_v + 2
                        else:
                            gain2 = gain1_u + gain1_v
                        
                        if gain2 > best_gain:
                            best_gain = gain2
                            best_v1, best_v2 = u, v
                
                if best_gain > 0:
                    # Apply 2-flip
                    u, v = best_v1, best_v2
                    old_u, old_v = labels[u], labels[v]
                    labels[u] = 1 - old_u
                    labels[v] = 1 - old_v
                    
                    # Update same[u] and same[v]
                    if labels[u] == 0:
                        same[u] = int(adj[u, labels == 0].sum())
                    else:
                        same[u] = int(adj[u, labels == 1].sum())
                    
                    if labels[v] == 0:
                        same[v] = int(adj[v, labels == 0].sum())
                    else:
                        same[v] = int(adj[v, labels == 1].sum())
                    
                    # Update neighbors
                    for w in neighbors[u]:
                        if w == v or w == u: continue
                        if labels[w] == old_u: same[w] -= 1
                        elif labels[w] == (1 - old_u): same[w] += 1
                    
                    for w in neighbors[v]:
                        if w == u or w == v: continue
                        if labels[w] == old_v: same[w] -= 1
                        elif labels[w] == (1 - old_v): same[w] += 1
                    
                    cut = compute_cut(labels)
                    improved = True
                    stall_count = 0
                    two_flip_count = 0
                    continue
            
            # Stuck: no improvement possible
            stall_count += 1
            if stall_count > max_stalls:
                break

        return labels, compute_cut(labels)

    def simulated_annealing(labels, time_limit):
        """
        Run simulated annealing with 1-flip and 2-flip moves to escape local minima.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut
        
        T_start = 8.0
        T_end = 0.02
        cooling_rate = 0.985
        T = T_start
        
        same = compute_same(labels)
        
        while T > T_end and time.time() < time_limit:
            # Choose between 1-flip and 2-flip moves
            use_2flip = np.random.rand() < 0.25
            
            if use_2flip and n >= 2:
                v1, v2 = np.random.choice(n, size=2, replace=False)
                if v1 == v2:
                    continue
                    
                gain_v1 = 2 * same[v1] - degrees[v1]
                gain_v2 = 2 * same[v2] - degrees[v2]
                
                # Compute combined gain considering edge between v1 and v2
                if adj[v1, v2] == 1:
                    if labels[v1] == labels[v2]:
                        combined_gain = gain_v1 + gain_v2 - 2
                    else:
                        combined_gain = gain_v1 + gain_v2 + 2
                else:
                    combined_gain = gain_v1 + gain_v2
                
                if combined_gain > 0 or (np.random.rand() < np.exp(-combined_gain / T)):
                    old1, old2 = labels[v1], labels[v2]
                    labels[v1] = 1 - old1
                    labels[v2] = 1 - old2
                    
                    if labels[v1] == 0:
                        same[v1] = int(adj[v1, labels == 0].sum())
                    else:
                        same[v1] = int(adj[v1, labels == 1].sum())
                    
                    if labels[v2] == 0:
                        same[v2] = int(adj[v2, labels == 0].sum())
                    else:
                        same[v2] = int(adj[v2, labels == 1].sum())
                    
                    for u in neighbors[v1]:
                        if u == v2 or u == v1:
                            continue
                        if labels[u] == old1:
                            same[u] -= 1
                        elif labels[u] == (1 - old1):
                            same[u] += 1
                    
                    for u in neighbors[v2]:
                        if u == v1 or u == v2:
                            continue
                        if labels[u] == old2:
                            same[u] -= 1
                        elif labels[u] == (1 - old2):
                            same[u] += 1
                    
                    cut = compute_cut(labels)
                    
                    if cut > best_cut:
                        best_cut = cut
                        best_labels = labels.copy()
            else:
                v = np.random.randint(0, n)
                gain = 2 * same[v] - degrees[v]
                
                if gain > 0 or (np.random.rand() < np.exp(-gain / T)):
                    old_label = labels[v]
                    labels[v] = 1 - old_label
                    new_label = labels[v]
                    
                    if new_label == 0:
                        same[v] = int(adj[v, labels == 0].sum())
                    else:
                        same[v] = int(adj[v, labels == 1].sum())
                    
                    for u in neighbors[v]:
                        if u == v:
                            continue
                        if labels[u] == old_label:
                            same[u] -= 1
                        elif labels[u] == new_label:
                            same[u] += 1
                        
                    cut = compute_cut(labels)
                    
                    if cut > best_cut:
                        best_cut = cut
                        best_labels = labels.copy()
            
            T *= cooling_rate

        return best_labels, best_cut

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # 1. Generate candidates using Goemans-Williamson Rounding
    candidates = []
    
    # Add a random start
    random_labels = rng.integers(0, 2, size=n, dtype=np.int8)
    candidates.append(random_labels)
    
    # Add a greedy start
    greedy_labels = np.zeros(n, dtype=np.int8)
    order = np.argsort(-degrees)
    for v in order:
        if len(neighbors[v]) == 0:
            continue
        count_1 = np.sum(greedy_labels[neighbors[v]] == 1)
        count_0 = np.sum(greedy_labels[neighbors[v]] == 0)
        if count_1 > count_0:
            greedy_labels[v] = 1
        else:
            greedy_labels[v] = 0
    candidates.append(greedy_labels)
    
    # GW Rounding with more eigenvectors
    try:
        adj_sym = (adj + adj.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(adj_sym)
        
        # Use more eigenvectors for richer geometric structure
        k = min(20, n)
        V = eigvecs[:, -k:]
        
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms
        
        num_rounds = 2000
        seen = set()
        
        for i in range(num_rounds):
            if time.time() > deadline - 25: # Leave more time for refinement
                break
                
            u = rng.standard_normal(k)
            u /= np.linalg.norm(u)
            
            proj = V @ u
            labels = (proj > 0).astype(np.int8)
            
            if np.all(labels == 0) or np.all(labels == 1):
                continue
                
            key = labels.tobytes()
            if key in seen:
                continue
            seen.add(key)
            
            candidates.append(labels)
            
            # Keep only top 15 candidates based on initial cut size
            if len(candidates) > 15:
                candidates.sort(key=lambda x: compute_cut(x), reverse=True)
                candidates = candidates[:15]
                
    except Exception:
        pass

    # 2. Refine candidates with Local Search
    for labels in candidates:
        if time.time() > deadline - 15:
            break
            
        ls_labels, ls_cut = local_search(labels, deadline - 15)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels

    # 3. Run Simulated Annealing on the best solution found
    if best_labels is not None and time.time() < deadline - 20:
        sa_labels, sa_cut = simulated_annealing(best_labels, deadline - 20)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
        
        # Run local search after SA to polish
        if time.time() < deadline - 5:
            ls_labels, ls_cut = local_search(best_labels, deadline - 5)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels
            
    # 4. Final Local Search polish
    if best_labels is not None and time.time() < deadline - 1:
        final_labels, final_cut = local_search(best_labels, deadline - 1)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)

# EVOLVE-BLOCK-END
