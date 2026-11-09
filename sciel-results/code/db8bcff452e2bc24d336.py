import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def _and9_fallback():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    def _mask_to_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def _has_independent_set(adj_mask, n, k, deadline):
        # Complement graph clique detection
        all_mask = (1 << n) - 1
        comp_mask = [all_mask ^ (1 << i) ^ adj_mask[i] for i in range(n)]
        
        # Order by degree in comp
        degs = [bin(comp_mask[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: degs[i], reverse=True)
        
        # Remap
        new_comp = [0] * n
        for new_idx, old_idx in enumerate(order):
            mask = 0
            for new_j, old_j in enumerate(order):
                if new_idx != new_j and (comp_mask[old_idx] >> old_j) & 1:
                    mask |= (1 << new_j)
            new_comp[new_idx] = mask
        
        def bnb(candidates, need):
            if time.time() > deadline:
                return True
            if need == 0:
                return True
            if candidates == 0:
                return False
            if bin(candidates).count('1') < need:
                return False
            
            # Pick first vertex in candidates (which has highest degree due to ordering)
            # Actually, we should pick the one with highest degree within candidates
            # But simple iteration is faster if ordering is good
            c = candidates
            while c:
                lb = c & (-c)
                idx = lb.bit_length() - 1
                c ^= lb
                
                remaining = candidates & (~((1 << (idx + 1)) - 1))
                new_candidates = remaining & new_comp[idx]
                
                if bnb(new_candidates, need - 1):
                    return True
            return False
        
        return bnb((1 << n) - 1, k)

    # Try to find a valid circulant graph for n down to 27
    for n in range(39, 26, -1):
        if time.time() > deadline - 10:
            break
            
        # Generate a random sum-free set S for Z_n
        # S is a subset of 1..n//2
        # We want S to be large to minimize alpha
        
        max_attempts = 100
        for _ in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Start with all possible edges
            S = set(range(1, n // 2 + 1))
            
            # Remove edges that form triangles
            # Triangle: a, b, a+b (mod n)
            # We iterate and remove
            changed = True
            while changed:
                changed = False
                to_remove = set()
                for a in list(S):
                    for b in list(S):
                        s = (a + b) % n
                        if s == 0:
                            continue
                        s = min(s, n - s)
                        if s in S:
                            # Triangle a, b, s
                            # Remove one, e.g., s
                            to_remove.add(s)
                            changed = True
                            break
                    if changed:
                        break
                S -= to_remove
            
            # Construct adj_mask
            adj_mask = [0] * n
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
            
            # Check independence
            if not _has_independent_set(adj_mask, n, 10, deadline - 5):
                return {'n': n, 'edges': _mask_to_edges(adj_mask, n)}
                
    return _and9_fallback()
