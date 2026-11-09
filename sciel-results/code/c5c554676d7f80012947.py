import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Increase restarts significantly due to speedup
        max_restarts = 500
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            # M[i] is a bitmask of neighbors of i
            M = [0] * target_n
            # D2[i] is a bitmask of vertices at distance 2 from i
            # D2[i] = union of M[w] for w in M[i]
            D2 = [0] * target_n
            deg = [0] * target_n
            
            # Precompute masks for quick access?
            # We need to iterate neighbors.
            # For bitset, getting neighbors is harder to iterate than list.
            # But for can_add, we don't need to iterate neighbors if we use D2.
            
            def can_add(u, v):
                if u == v:
                    return False
                if M[u] & (1 << v):
                    return False
                
                # Check common neighbors
                common = M[u] & M[v]
                if common.bit_count() > 1:
                    return False
                
                # Check C4 creation:
                # Path u-x-w-v where x in N(u), w in N(v), x~w.
                # This is equivalent to N(u) intersecting (Union of N(w) for w in N(v)).
                # Union of N(w) for w in N(v) is D2[v].
                # But D2[v] includes v's neighbors? No, D2 is distance 2.
                # If x in N(u) and x in D2[v], then there is w in N(v) such that x in N(w).
                # Then u-x-w-v is a path of length 3.
                # So we need (M[u] & D2[v]) == 0.
                # And (M[v] & D2[u]) == 0.
                
                if (M[u] & D2[v]) != 0:
                    return False
                if (M[v] & D2[u]) != 0:
                    return False
                
                return True

            def add_edge(u, v):
                # Update M
                M[u] |= (1 << v)
                M[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                # Update D2
                # 1. v becomes distance 2 from all neighbors of u (except v itself, but v not in N(u) initially)
                #    So for all x in N(u), D2[x] gains v.
                #    And D2[v] gains all x in N(u).
                # 2. u becomes distance 2 from all neighbors of v.
                #    So for all y in N(v), D2[y] gains u.
                #    And D2[u] gains all y in N(v).
                
                # Update D2[v] with N(u)
                D2[v] |= M[u]
                # Update D2[u] with N(v)
                D2[u] |= M[v]
                
                # Update D2[x] for x in N(u): add v
                # Iterate through bits of M[u]
                mask_u = M[u]
                while mask_u:
                    lsb = mask_u & -mask_u
                    x = lsb.bit_length() - 1
                    D2[x] |= (1 << v)
                    mask_u ^= lsb
                    
                # Update D2[y] for y in N(v): add u
                mask_v = M[v]
                while mask_v:
                    lsb = mask_v & -mask_v
                    y = lsb.bit_length() - 1
                    D2[y] |= (1 << u)
                    mask_v ^= lsb

            def remove_edge(u, v):
                # This is complex to do with bitsets correctly without recounting D2.
                # Since we are doing greedy addition, we might not need removal if we are smart.
                # But if we get stuck, we need to restart or backtrack.
                # Given the speedup, restarting is likely cheaper than implementing correct removal.
                # So we will just restart if stuck.
                pass

            success = True
            stuck = False
            
            for step in range(5000): # Max steps
                if time.time() > deadline:
                    success = False
                    break
                    
                # Check if we are done
                if min(deg) >= min_deg_needed:
                    success = True
                    break

                # Find vertex with min degree
                min_deg = min(deg)
                
                low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
                if not low_vertices:
                    success = True
                    break
                    
                u = random.choice(low_vertices)
                
                # Find a valid v to connect to u
                candidates = []
                for v in range(target_n):
                    if v == u:
                        continue
                    if can_add(u, v):
                        candidates.append(v)
                        
                if not candidates:
                    stuck = True
                    break
                            
                v = random.choice(candidates)
                add_edge(u, v)
                
            if stuck:
                continue

            if success:
                # Final verification
                # Recalculate D2 or just check C4 via common neighbors
                # C4-free iff for all u<v, common(u,v) <= 1
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (M[i] & M[j]).bit_count() > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free and min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        mask = M[i]
                        while mask:
                            lsb = mask & -mask
                            j = lsb.bit_length() - 1
                            if i < j:
                                final_edges.append([i, j])
                            mask ^= lsb
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
