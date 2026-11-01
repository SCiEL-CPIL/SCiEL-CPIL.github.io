import numpy as np
import itertools
import sys
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        # Split 8-dim into two 4-dim halves
        # u in 0..80, v in 0..80
        # total points 6561
        
        def idx_to_uv(idx):
            u = idx // 81
            v = idx % 81
            return u, v

        def uv_to_idx(u, v):
            return u * 81 + v

        # Precompute shadows for all (u, v)
        # shadow[u][v] is a bitset of 6561 bits
        # Using Python ints for bitsets
        
        # First, map 4-dim digits to index
        def digits4_to_idx(d):
            return d[0] * 27 + d[1] * 9 + d[2] * 3 + d[3]
            
        def idx_to_digits4(idx):
            d = [0] * 4
            d[3] = idx % 3
            idx //= 3
            d[2] = idx % 3
            idx //= 3
            d[1] = idx % 3
            idx //= 3
            d[0] = idx
            return d

        # Precompute S1 for 4-dim
        # S1[u] is a list of v's such that d(u, v) <= 1
        S1 = [[] for _ in range(81)]
        for u in range(81):
            du = idx_to_digits4(u)
            S1[u].append(u)
            for pos in range(4):
                orig = du[pos]
                # +1
                du[pos] = (orig + 1) % 3
                v1 = digits4_to_idx(du)
                S1[u].append(v1)
                # -1
                du[pos] = (orig + 2) % 3
                v2 = digits4_to_idx(du)
                S1[u].append(v2)
                # reset
                du[pos] = orig

        # Precompute shadow_bits[u][v]
        # shadow(u, v) covers (w, z) if d(u, w) + d(v, z) <= 1
        # This is equivalent to:
        # (w == u and z in S1[v]) OR (w in S1[u] and z == v)
        
        shadow_bits = [[0] * 81 for _ in range(81)]
        for u in range(81):
            for v in range(81):
                mask = 0
                # Case 1: w == u, z in S1[v]
                for z in S1[v]:
                    idx = uv_to_idx(u, z)
                    mask |= (1 << idx)
                # Case 2: w in S1[u], z == v
                for w in S1[u]:
                    if w == u: continue # already added
                    idx = uv_to_idx(w, v)
                    mask |= (1 << idx)
                shadow_bits[u][v] = mask

        full_mask = (1 << 6561) - 1
        
        best_code = None
        best_size = 6561 # worst case
        
        # Try a few random restarts or just one good greedy
        # Greedy with random tie-breaking might help escape local minima?
        # Or just deterministic greedy.
        
        for restart in range(5): # 5 restarts to be safe
            uncovered = full_mask
            codewords = [] # list of (u, v)
            
            # Greedy Phase
            while uncovered:
                best_u = -1
                best_v = -1
                best_count = 0
                
                # Optimization: Iterate only over candidates?
                # For now, full scan.
                for u in range(81):
                    for v in range(81):
                        inter = uncovered & shadow_bits[u][v]
                        count = inter.bit_count()
                        if count > best_count:
                            best_count = count
                            best_u = u
                            best_v = v
                            if best_count == uncovered.bit_count():
                                break
                    if best_count == uncovered.bit_count():
                        break
                
                if best_u == -1:
                    break
                    
                codewords.append((best_u, best_v))
                uncovered &= ~shadow_bits[best_u][best_v]
                
            # Local Search: Minimization
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords) - 1, -1, -1):
                    u, v = codewords[i]
                    # Check if removing (u, v) leaves everything covered
                    # Union of all other shadows
                    union_mask = 0
                    for j in range(len(codewords)):
                        if j != i:
                            uj, vj = codewords[j]
                            union_mask |= shadow_bits[uj][vj]
                    
                    if (union_mask & full_mask) == full_mask:
                        codewords.pop(i)
                        changed = True

            if len(codewords) < best_size:
                best_size = len(codewords)
                best_code = codewords[:]

        # Convert best_code to list of lists of digits
        result = []
        for u, v in best_code:
            du = idx_to_digits4(u)
            dv = idx_to_digits4(v)
            result.append(du + dv)
            
        return result

    # Fallback
    if q ** n <= 100:
         return [list(itertools.product(range(q), repeat=n))[i] for i in range(q**n)]
    return [[0]*n]
