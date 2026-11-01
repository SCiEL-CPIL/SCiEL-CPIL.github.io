import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate all candidate vectors with norm^2 = 4
    # Type 1: Weight 4 (four +/-1)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Type 2: Weight 1 (one +/-2)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    all_vectors = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_vectors)
    
    # 2. Compute conflict matrix
    # Conflict if dot > 2
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    best_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    def greedy_min_degree():
        """Greedy selection of minimum degree nodes."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            # Remove chosen and its neighbors from 'alive'
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Decrease degree by number of neighbors removed
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set.
        counts[v] is the number of neighbors of v currently in in_set.
        """
        added = 0
        # Iterate in random order to avoid bias, or just standard order
        indices = rng.permutation(N)
        for v in indices:
            if not in_set[v] and counts[v] == 0:
                in_set[v] = True
                added += 1
                # Update counts for neighbors of v
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        """
        Local search: try swaps and fills.
        """
        in_set = in_set.copy()
        # Calculate current conflict counts (should be 0 if valid, but let's be sure)
        counts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        # Ensure validity first (remove nodes with conflicts)
        while True:
            conflicted = np.where((counts > 0) & in_set)[0]
            if len(conflicted) == 0:
                break
            # Remove one node from the conflicted set (random or max degree)
            # To be fast, just remove the first one found
            v = conflicted[0]
            in_set[v] = False
            for u in neighbors[v]:
                counts[u] -= 1
            # Note: counts for v itself doesn't matter since it's out, 
            # but neighbors of v that were in set had their counts reduced.
            # Wait, if v had conflicts, removing v reduces counts of its neighbors.
            # The neighbors of v that are in the set are the ones whose counts decrease.
            # My update `counts[neighbors[v]] -= 1` is correct for neighbors of v.
            # But what about v's own count? It's not used since in_set[v] is False.
        
        # Now in_set is a valid independent set.
        # Try to fill
        greedy_fill(in_set, counts)
        
        # Try swaps: Remove one node, see if we can add more than 1 new node
        # To be efficient, we only try this for a subset of nodes or iterate a few times
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Re-calculate counts for current set
            counts = np.zeros(N, dtype=np.int32)
            for i in np.where(in_set)[0]:
                counts[neighbors[i]] += 1
            
            # Pick a random node from the set to try removing
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) == 0:
                break
            
            # Try a limited number of swaps per pass
            swap_attempts = min(len(in_set_indices), 50)
            for _ in range(swap_attempts):
                v = in_set_indices[rng.integers(len(in_set_indices))]
                # Temporarily remove v
                in_set[v] = False
                for u in neighbors[v]:
                    counts[u] -= 1
                
                # Try to fill
                added = greedy_fill(in_set, counts)
                
                if added > 1:
                    # Net gain in size
                    improved = True
                    break # Restart the swap loop with new set
                else:
                    # Revert: put v back
                    in_set[v] = True
                    for u in neighbors[v]:
                        counts[u] += 1
                    
        return in_set

    # Main Loop
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
            
        # Early exit if we hit the theoretical max (165)
        if best_size >= 165:
            break

    # Final result
    return all_vectors[best_set]

if __name__ == "__main__":
    # Test run
    start = time.time()
    res = construct()
    print(f"Size: {len(res)}, Time: {time.time() - start:.2f}s")
