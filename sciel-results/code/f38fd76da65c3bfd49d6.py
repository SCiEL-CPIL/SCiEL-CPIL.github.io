def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 10.0  # seconds before deadline to stop
    
    def make_orientation(n, bits):
        """Convert bit list to orientation list in lexicographic order."""
        orientation = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                orientation.append(bits[idx])
                idx += 1
        return orientation
    
    def check_validity(n, orientation):
        """Check if every 4-set has a common dominator outside the set."""
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        # For each 4-set, check if there exists a vertex outside that dominates all 4
        # A vertex v dominates set S if adj[v, u] = 1 for all u in S
        # We need: for all S with |S|=4, exists v not in S such that adj[v, u]=1 for all u in S
        
        # Precompute: for each vertex v, which vertices it dominates
        # dom_mask[v] is a bitmask of vertices dominated by v
        
        # For efficiency, we'll check by iterating through 4-sets
        # But this is O(n^4) which is too slow for large n
        # Instead, we can use a different approach:
        # For each vertex v, let D(v) = {u : v->u}
        # We need: for every 4-set S, exists v not in S such that S ⊆ D(v)
        # Equivalently: the collection {D(v) \ {v} : v in V} covers all 4-subsets
        
        # Let's use a sampling approach for speed, but we need exactness
        # For n up to ~50, O(n^4) might be feasible with numpy
        # Let's try a vectorized approach
        
        # For each 4-set (a,b,c,d), we need exists v not in {a,b,c,d} with adj[v,a]=adj[v,b]=adj[v,c]=adj[v,d]=1
        # This is equivalent to: min over v not in {a,b,c,d} of (adj[v,a]+adj[v,b]+adj[v,c]+adj[v,d]) <= 0
        
        # Precompute for each v, the out-neighborhood as a set
        # Actually, let's just do the O(n^4) check but with early termination
        
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if any vertex outside {a,b,c,d} dominates all four
                        found = False
                        for v in range(n):
                            if v == a or v == b or v == c or v == d:
                                continue
                            if adj[v, a] and adj[v, b] and adj[v, c] and adj[v, d]:
                                found = True
                                break
                        if not found:
                            return False
        return True
    
    def check_validity_fast(n, orientation):
        """Faster validity check using numpy."""
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        # For each 4-set, we need a vertex v outside the set that dominates all 4
        # Use numpy to speed up the inner loop
        # Precompute column sums won't help directly
        
        # Alternative: for each vertex v, compute the set of 4-sets it dominates
        # A vertex v dominates a 4-set S iff S ⊆ out_neighbors(v) \ {v}
        # So the number of 4-sets dominated by v is C(outdeg(v), 4)
        # But we need coverage, not just count
        
        # Let's try: for each 4-set, check if any vertex outside dominates it
        # We can vectorize over the 4th index for fixed (a,b,c)
        
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    # For each d > c, check if exists v not in {a,b,c,d} with adj[v,a]=adj[v,b]=adj[v,c]=adj[v,d]=1
                    # Let M = {v : adj[v,a]=1 and adj[v,b]=1 and adj[v,c]=1 and v != a,b,c}
                    # For d, we need d not in M and exists v in M with adj[v,d]=1
                    # Actually, we need v not in {a,b,c,d}, so v in M and v != d and adj[v,d]=1
                    
                    mask_abc = (adj[:, a] == 1) & (adj[:, b] == 1) & (adj[:, c] == 1)
                    mask_abc[a] = False
                    mask_abc[b] = False
                    mask_abc[c] = False
                    
                    for d in range(c + 1, n):
                        # Need v in mask_abc, v != d, adj[v,d]=1
                        # Check if any v in mask_abc with v != d and adj[v,d]=1
                        # If mask_abc[d] is True, we can still use v=d? No, v must be outside {a,b,c,d}
                        # So v must be in mask_abc and v != d
                        # And adj[v,d]=1
                        candidates = mask_abc.copy()
                        candidates[d] = False
                        if np.any(candidates & (adj[:, d] == 1)):
                            continue
                        else:
                            return False
        return True
    
    def repair(n, orientation, max_time=5.0):
        """Try to repair the tournament by flipping edges."""
        start = time.time()
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        def count_bad_4sets(adj):
            """Count the number of 4-sets without a common dominator."""
            bad = 0
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        for d in range(c + 1, n):
                            found = False
                            for v in range(n):
                                if v == a or v == b or v == c or v == d:
                                    continue
                                if adj[v, a] and adj[v, b] and adj[v, c] and adj[v, d]:
                                    found = True
                                    break
                            if not found:
                                bad += 1
            return bad
        
        # Start with current state
        current_bad = count_bad_4sets(adj)
        if current_bad == 0:
            return orientation, True
        
        best_orientation = orientation[:]
        best_bad = current_bad
        
        # Local search: flip edges to reduce bad 4-sets
        for iteration in range(10000):
            if time.time() - start > max_time:
                break
            if best_bad == 0:
                break
            
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Flip edge (i,j)
            old_val = adj[i, j]
            adj[i, j] = 1 - old_val
            adj[j, i] = 1 - adj[i, j]
            
            new_bad = count_bad_4sets(adj)
            if new_bad < best_bad:
                best_bad = new_bad
                # Update best orientation
                idx2 = 0
                for ii in range(n):
                    for jj in range(ii + 1, n):
                        if adj[ii, jj] == 1:
                            best_orientation[idx2] = 0
                        else:
                            best_orientation[idx2] = 1
                        idx2 += 1
            else:
                # Revert
                adj[i, j] = old_val
                adj[j, i] = 1 - old_val
        
        return best_orientation, (best_bad == 0)
    
    # Try n from 31 upward
    for n in range(31, 91):
        if time.time() + safety_margin > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() + safety_margin > deadline:
                break
            
            # Random orientation
            orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            
            # Check if valid
            if check_validity_fast(n, orientation):
                return {'n': n, 'orientation': orientation}
            
            # Try to repair
            repaired, is_valid = repair(n, orientation, max_time=3.0)
            if is_valid:
                return {'n': n, 'orientation': repaired}
        
        # If we can't find a valid tournament for this n, try larger n
        # But we should be careful about time
    
    # Fallback: use Paley tournament on 67 vertices (known valid)
    p = 67
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
