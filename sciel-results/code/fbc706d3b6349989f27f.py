import math
import time
from math import gcd
from itertools import combinations

def construct():
    """
    Find a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all
    distinct pairs, maximizing cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Every element in the set must have gcd > 1 with 255
    # So every element must be divisible by 3, 5, or 17
    candidates = [x for x in range(1, 256) if gcd(x, 255) > 1]
    n = len(candidates)  # 127
    
    # Build adjacency: adj[i] is set of j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum clique via branch and bound with coloring-based upper bound
    # Order vertices by degree descending for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Remap
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    
    adj2 = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            adj2[remap[i]].add(remap[j])
    
    # Greedy coloring for upper bound (Tomita's algorithm)
    def color_sort(P):
        """Color vertices in P greedily, return list of (vertex, color) sorted by color."""
        P_list = list(P)
        colors = {}
        color = 0
        # Sort by degree descending for better coloring
        P_list.sort(key=lambda v: -len(adj2[v] & P))
        for v in P_list:
            # Assign smallest available color
            used = set()
            for u in adj2[v]:
                if u in P and u in colors:
                    used.add(colors[u])
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            color = max(color, c)
        # Return sorted by color
        return sorted(colors.items(), key=lambda x: x[1])
    
    best_clique = []
    best_size = 0
    
    def expand(R, P):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        # Color sort for upper bound
        colored = color_sort(P)
        
        # Process vertices in reverse color order (higher color = tighter bound)
        for v, c in reversed(colored):
            if time.time() > deadline:
                return
            
            if len(R) + c <= best_size:
                break  # No point continuing, colors are non-increasing
            
            # v must be in P
            if v not in P:
                continue
                
            new_R = R | {v}
            new_P = P & adj2[v]
            
            expand(new_R, new_P)
            P.remove(v)
    
    # Start with all vertices
    expand(set(), set(range(n)))
    
    # Map back to original values
    result_indices = [order[i] for i in best_clique]
    result = [candidates[i] for i in result_indices]
    
    # Ensure 255 is in the result
    if 255 not in result:
        idx_255 = candidates.index(255)
        if remap[idx_255] not in best_clique:
            # Try to add 255
            if all(gcd(255, candidates[i]) > 1 for i in result_indices):
                result.append(255)
            else:
                # Remove incompatible and add 255
                new_result = []
                for val in result:
                    if gcd(255, val) > 1:
                        new_result.append(val)
                new_result.append(255)
                result = new_result
    
    # Final verification and cleanup
    result = list(set(result))
    if 255 not in result:
        result.append(255)
    
    # Verify all pairs have gcd > 1
    valid_result = []
    for val in result:
        ok = True
        for other in valid_result:
            if gcd(val, other) == 1:
                ok = False
                break
        if ok:
            valid_result.append(val)
    
    if 255 not in valid_result:
        valid_result.append(255)
    
    return valid_result
