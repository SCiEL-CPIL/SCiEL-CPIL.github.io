import time
from math import gcd

def construct():
    """Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs. Maximize cardinality."""
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Candidate pool: numbers in [1, 255] that share a factor with 255 (3, 5, 17)
    # Any valid set containing 255 must only contain numbers sharing a factor with 255.
    # If x is in the set, gcd(x, 255) > 1.
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
        
    n_cand = len(candidates)
    
    # Map value to index in candidates for faster lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # Precompute gcd matrix (adjacency)
    # adj[i][j] is True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        vi = candidates[i]
        for j in range(i + 1, n_cand):
            vj = candidates[j]
            if gcd(vi, vj) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    def is_valid_set(indices):
        """Check if the set of indices forms a clique."""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if not adj[indices[i]][indices[j]]:
                    return False
        return True

    def get_conflicts(idx, current_indices):
        """Return list of indices in current_indices that conflict with idx."""
        conflicts = []
        for y in current_indices:
            if not adj[idx][y]:
                conflicts.append(y)
        return conflicts

    best_set = []
    best_size = 0
    
    # Initial seeds
    # 1. All multiples of 3
    seed1 = [i for i, v in enumerate(candidates) if v % 3 == 0]
    # 2. All multiples of 5
    seed2 = [i for i, v in enumerate(candidates) if v % 5 == 0]
    # 3. All multiples of 17
    seed3 = [i for i, v in enumerate(candidates) if v % 17 == 0]
    # 4. Just 255
    seed4 = [idx_255]
    
    seeds = [seed1, seed2, seed3, seed4]
    
    # Local search function
    def local_search(initial_indices, max_iters=10000):
        current = set(initial_indices)
        # Ensure 255 is present
        if idx_255 not in current:
            # If 255 conflicts, we can't use this seed directly, but our seeds are valid cliques
            # Actually, if seed doesn't contain 255, we should discard or force it?
            # The task requires 255.
            # Our seeds 1,2,3 contain 255. Seed 4 is 255.
            pass
            
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding any single node
            added = False
            for x in range(n_cand):
                if x in current:
                    continue
                # Check if x is compatible with all in current
                # Optimization: check conflicts
                conflicts = [y for y in current if not adj[x][y]]
                if len(conflicts) == 0:
                    current.add(x)
                    improved = True
                    added = True
                    break # Restart loop with new set
            
            if not added:
                # Try removing one node to allow adding others?
                # This is a heuristic. If we remove y, we might add x1, x2...
                # For simplicity, we stick to single-add for now, but we can try "swap"
                # Remove y, add x if |current| - 1 + 1 > |current| is false.
                # So single swap is never an improvement in size.
                # We need to remove y and add x1, x2 such that they are compatible with each other and current\{y}.
                # This is expensive.
                # Alternative: Random restarts or different seeds are more effective.
                break
                
        return list(current)

    # Run local search on seeds
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        result = local_search(seed)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    # Convert indices back to values
    best_values = [candidates[i] for i in best_set]
    
    # If best_set is empty or doesn't contain 255 (shouldn't happen with seeds), fallback
    if not best_values or 255 not in best_values:
        # Fallback to multiples of 3
        best_values = [v for v in candidates if v % 3 == 0]
        if 255 not in best_values:
            best_values.append(255)
            
    return sorted(best_values)
