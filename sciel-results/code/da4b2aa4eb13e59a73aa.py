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
        # Vectorized: same[v] = sum of adj[v,u] where labels[u] == labels[v]
        # Use matrix multiplication: same = adj @ (labels == 0) for label 0 vertices, etc.
        label0_mask = (labels == 0).astype(np.int8)
        label1_mask = (labels == 1).astype(np.int8)
        same = np.zeros(n, dtype=np.int32)
        # For vertices with label 0: count neighbors with label 0
        same[label0_mask == 1] = (adj @ label0_mask)[label0_mask == 1]
        # For vertices with label 1: count neighbors with label 1
        same[label1_mask == 1] = (adj @ label1_mask)[label1_mask == 1]
        return same

    def local_search(labels, time_limit):
        """
        Run local search with 1-flip and 2-flip moves.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        same = compute_same(labels)
        
        # Track which vertices are in the heap to avoid duplicates
        in_heap = np.zeros(n, dtype=bool)
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))
                in_heap[v] = True

        improved = True
        stall_count = 0
        while improved and time.time() < time_limit:
            improved = False
            stall_count += 1
            
            # Try 1-flip moves first
            while heap:
                neg_gain, v = heapq.heappop(heap)
                in_heap[v] = False
                gain = -neg_gain

                # Verify gain is still valid
                current_gain = 2 * same[v] - degrees[v]
                if current_gain != gain or current_gain <= 0:
                    continue

                # Flip vertex v
                old_label = labels[v]
                labels[v] = 1 - old_label
                new_label = labels[v]

                # Update same[v] using vectorized computation for this vertex only
                if new_label == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())

                # Update neighbors of v
                for u in neighbors[v]:
                    if u == v:
                        continue
                    if labels[u] == old_label:
                        same[u] -= 1
                    elif labels[u] == new_label:
                        same[u] += 1
                    
                    # Check if u now has positive gain and not already in heap
                    u_gain = 2 * same[u] - degrees[u]
                    if u_gain > 0 and not in_heap[u]:
                        heapq.heappush(heap, (-u_gain, u))
                        in_heap[u] = True

                cut += current_gain
                improved = True
                stall_count = 0
                break
            
            # If no 1-flip improvement, try a 2-flip move
            if not improved and time.time() < time_limit:
                if stall_count >= 10:
                    # Try random 2-flip move
                    if n >= 2:
                        v1, v2 = np.random.choice(n, size=2, replace=False)
                        if v1 != v2:
                            # Calculate gain of flipping both v1 and v2
                            gain_v1 = 2 * same[v1] - degrees[v1]
                            gain_v2 = 2 * same[v2] - degrees[v2]
                            combined_gain = gain_v1 + gain_v2
                            
                            if combined_gain > 0:
                                # Perform the 2-flip
                                old1, old2 = labels[v1], labels[v2]
                                labels[v1] = 1 - old1
                                labels[v2] = 1 - old2
                                
                                # Update same for v1
                                if labels[v1] == 0:
                                    same[v1] = int(adj[v1, labels == 0].sum())
                                else:
                                    same[v1] = int(adj[v1, labels == 1].sum())
                                
                                # Update same for v2
                                if labels[v2] == 0:
                                    same[v2] = int(adj[v2, labels == 0].sum())
                                else:
                                    same[v2] = int(adj[v2, labels == 1].sum())
                                
                                # Update neighbors of v1 (excluding v2 which we handle separately)
                                for u in neighbors[v1]:
                                    if u == v2 or u == v1:
                                        continue
                                    if labels[u] == old1:
                                        same[u] -= 1
                                    elif labels[u] == (1 - old1):
                                        same[u] += 1
                                    u_gain = 2 * same[u] - degrees[u]
                                    if u_gain > 0 and not in_heap[u]:
                                        heapq.heappush(heap, (-u_gain, u))
                                        in_heap[u] = True
                                
                                # Update neighbors of v2 (excluding v1)
                                for u in neighbors[v2]:
                                    if u == v1 or u == v2:
                                        continue
                                    if labels[u] == old2:
                                        same[u] -= 1
                                    elif labels[u] == (1 - old2):
                                        same[u] += 1
                                    u_gain = 2 * same[u] - degrees[u]
                                    if u_gain > 0 and not in_heap[u]:
                                        heapq.heappush(heap, (-u_gain, u))
                                        in_heap[u] = True
                                
                                cut += combined_gain
                                improved = True
                                stall_count = 0
                    # Reset stall counter periodically to avoid infinite 2-flip loops
                    if stall_count > 100:
                        stall_count = 0

        return labels, cut

    def simulated_annealing(labels, time_limit):
        """
        Run simulated annealing to escape local minima.
        Returns (labels, cut_size).
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut
        
        T_start = 8.0
        T_end = 0.02
        cooling_rate = 0.98
        T = T_start
        
        same = compute_same(labels)
        
        while T > T_end and time.time() < time_limit:
            # Occasionally do a 2-flip move (higher probability at higher T)
            two_flip_prob = 0.15 + 0.1 * (T / T_start)
            if np.random.rand() < two_flip_prob and n >= 2:
                v1, v2 = np.random.choice(n, size=2, replace=False)
                if v1 != v2:
                    gain_v1 = 2 * same[v1] - degrees[v1]
                    gain_v2 = 2 * same[v2] - degrees[v2]
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
                        
                        cut += combined_gain
                        
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
                        
                    cut += gain
                    
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
    
    # Add multiple random starts
    for _ in range(3):
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
        k = min(25, n)
        V = eigvecs[:, -k:]
        
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms
        
        num_rounds = 3000
        seen = set()
        
        for i in range(num_rounds):
            if time.time() > deadline - 20: # Leave more time for refinement
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
            
            # Keep only top 20 candidates based on initial cut size
            if len(candidates) > 20:
                candidates.sort(key=lambda x: compute_cut(x), reverse=True)
                candidates = candidates[:20]
                
    except Exception:
        pass

    # 2. Refine candidates with Local Search
    for labels in candidates:
        if time.time() > deadline - 12:
            break
            
        ls_labels, ls_cut = local_search(labels, deadline - 12)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels

    # 3. Run Simulated Annealing on the best solution found
    if best_labels is not None and time.time() < deadline - 8:
        sa_labels, sa_cut = simulated_annealing(best_labels, deadline - 8)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
            
    # 4. Final Local Search polish (multiple passes)
    if best_labels is not None:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            final_labels, final_cut = local_search(best_labels, deadline - 2)
            if final_cut > best_cut:
                best_cut = final_cut
                best_labels = final_labels
            else:
                break

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)

# EVOLVE-BLOCK-END
