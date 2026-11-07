"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = t0 + 55.0  # safety margin before 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute validity for all 2^n masks
    full_mask = (1 << n) - 1
    valid_part = [False] * (1 << n)
    for mask in range(1, 1 << n):
        # Check if mask is a clique
        is_cl = True
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            others = mask ^ lsb
            if (adj[v] & others) != others:
                is_cl = False
                break
        if is_cl:
            valid_part[mask] = True
        else:
            # Check if mask is an independent set
            is_ind = True
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                others = mask ^ lsb
                if (adj[v] & others) != 0:
                    is_ind = False
                    break
            if is_ind:
                valid_part[mask] = True
    
    # Greedy fallback
    def greedy():
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Backtracking to find partition with exactly K parts
    def solve_k(K):
        def dfs(parts, unassigned):
            if time.time() > DEADLINE:
                return None
            
            if unassigned == 0:
                # All vertices assigned
                result = []
                for p in parts:
                    if p != 0:
                        lst = []
                        m = p
                        while m:
                            lsb = m & (-m)
                            lst.append(lsb.bit_length() - 1)
                            m ^= lsb
                        lst.sort()
                        result.append(lst)
                return result
            
            # Pick the most constrained unassigned vertex:
            # the one with fewest valid placements
            best_v = -1
            best_count = K + 1
            m = unassigned
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                v_bit = 1 << v
                count = 0
                tried = set()
                for i in range(K):
                    if parts[i] in tried:
                        continue
                    tried.add(parts[i])
                    new_part = parts[i] | v_bit
                    if valid_part[new_part]:
                        count += 1
                if count < best_count:
                    best_count = count
                    best_v = v
                    if count <= 1:
                        break
            
            if best_v == -1 or best_count == 0:
                return None
            
            v = best_v
            new_unassigned = unassigned ^ (1 << v)
            v_bit = 1 << v
            
            tried_masks = set()
            for i in range(K):
                # Symmetry breaking: skip if same mask already tried
                if parts[i] in tried_masks:
                    continue
                tried_masks.add(parts[i])
                
                new_part = parts[i] | v_bit
                if not valid_part[new_part]:
                    continue
                
                # Recurse
                new_parts = list(parts)
                new_parts[i] = new_part
                result = dfs(new_parts, new_unassigned)
                if result is not None:
                    return result
            
            return None
        
        # Initial state: K empty parts
        initial_parts = [0] * K
        return dfs(initial_parts, full_mask)
    
    # Try increasing K
    best_result = None
    for K in range(2, n + 1):
        if time.time() > DEADLINE:
            break
        result = solve_k(K)
        if result is not None:
            # Verify: all vertices present exactly once
            all_vertices = []
            for part in result:
                all_vertices.extend(part)
            if sorted(all_vertices) == list(range(n)):
                # Verify each part is non-empty and valid
                valid = True
                for part in result:
                    if len(part) == 0:
                        valid = False
                        break
                    mask = 0
                    for v in part:
                        mask |= (1 << v)
                    if not valid_part[mask]:
                        valid = False
                        break
                if valid:
                    best_result = result
                    break
    
    if best_result is None:
        best_result = greedy()
    
    # Final verification
    all_vertices = []
    for part in best_result:
        all_vertices.extend(part)
    if sorted(all_vertices) != list(range(n)):
        best_result = greedy()
    
    # Ensure types are correct
    result = []
    for part in best_result:
        result.append([int(v) for v in part])
    return result
# EVOLVE-BLOCK-END
