import time
import random
from itertools import combinations

def construct():
        start_time = time.time()
        deadline = 115
        
        def is_non_2_colorable(n, edges):
            # Fast bitmask solver
            m = len(edges)
            if m == 0: return False
            
            edge_masks = []
            for e in edges:
                mask = 0
                for v in e: mask |= (1 << v)
                edge_masks.append(mask)
            
            # Propagation state: mask0, mask1
            # We need to handle the search space.
            # To be fast, we can use a recursive DFS with bitmasks.
            
            def dfs(mask0, mask1):
                # Propagate
                changed = True
                while changed:
                    changed = False
                    for em in edge_masks:
                        c0 = (em & mask0).bit_count()
                        c1 = (em & mask1).bit_count()
                        if c0 == 5 or c1 == 5:
                            return False
                        if c0 == 4:
                            unassigned = em & ~(mask0 | mask1)
                            if unassigned:
                                # Must be 1
                                if mask1 & unassigned:
                                    # Conflict? No, if unassigned, it's not in mask1.
                                    # Wait, if unassigned has multiple bits, c0=4 implies 1 unassigned.
                                    pass
                                mask1 |= unassigned
                                changed = True
                        elif c1 == 4:
                            unassigned = em & ~(mask0 | mask1)
                            if unassigned:
                                mask0 |= unassigned
                                changed = True
                            
                unassigned = ((1 << n) - 1) & ~(mask0 | mask1)
                if unassigned == 0:
                    return True # Found coloring
                
                # Heuristic: pick vertex with most constraints?
                # Simple: first bit
                v_bit = unassigned & -unassigned
                v = v_bit.bit_length() - 1
                
                if dfs(mask0 | v_bit, mask1): return True
                if dfs(mask0, mask1 | v_bit): return True
                return False
            
            return not dfs(0, 0)

        def minimize(n, edges, time_limit):
            current_edges = [list(e) for e in edges]
            improved = True
            while improved:
                improved = False
                random.shuffle(current_edges)
                for i in range(len(current_edges)):
                    if time.time() > time_limit: return current_edges
                    test_edges = [e for j, e in enumerate(current_edges) if j != i]
                    if is_non_2_colorable(n, test_edges):
                        current_edges = test_edges
                        improved = True
                        break
            return current_edges

        best_edges = None
        best_count = 200
        
        # 1. Try minimizing K_9^5
        k9_edges = [list(e) for e in combinations(range(9), 5)]
        if is_non_2_colorable(9, k9_edges):
            min_k9 = minimize(9, k9_edges, deadline - 10)
            if len(min_k9) < best_count:
                best_count = len(min_k9)
                best_edges = min_k9
        
        # 2. Try random subsets of K_10^5
        all_k10 = list(combinations(range(10), 5))
        for _ in range(20): # 20 attempts
            if time.time() > deadline - 5: break
            random.shuffle(all_k10)
            # Try sizes around 50-60
            for m in [55, 50, 60]:
                candidate = [list(e) for e in all_k10[:m]]
                if is_non_2_colorable(10, candidate):
                    min_cand = minimize(10, candidate, deadline - 2)
                    if len(min_cand) < best_count:
                        best_count = len(min_cand)
                        best_edges = min_cand
                    break # If found one, minimize it, don't try larger m for this shuffle
        
        # 3. Try random subsets of K_11^5
        all_k11 = list(combinations(range(11), 5))
        for _ in range(10):
            if time.time() > deadline - 2: break
            random.shuffle(all_k11)
            for m in [55, 50]:
                candidate = [list(e) for e in all_k11[:m]]
                if is_non_2_colorable(11, candidate):
                    min_cand = minimize(11, candidate, deadline - 1)
                    if len(min_cand) < best_count:
                        best_count = len(min_cand)
                        best_edges = min_cand
                    break

        if best_edges:
            return {'n': len(best_edges[0][0]) if False else (9 if best_edges == min_k9 else (10 if len(set(v for e in best_edges for v in e)) <= 10 else 11)), 'edges': best_edges}
            # Need to track n correctly
        # Fallback
        return {'n': 9, 'edges': k9_edges}
