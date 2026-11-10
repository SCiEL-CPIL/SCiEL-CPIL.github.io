"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct():
    n = 256
    
    # Precompute all 4-term APs
    four_aps = []
    four_aps_by_vertex = {i: [] for i in range(1, n+1)}
    for d in range(1, (n-1)//3 + 1):
        for a in range(1, n - 3*d + 1):
            ap = (a, a+d, a+2*d, a+3*d)
            idx = len(four_aps)
            four_aps.append(ap)
            for v in ap:
                four_aps_by_vertex[v].append(idx)
                
    # Precompute weights: number of 3-APs each vertex participates in
    weights = [0] * (n + 1)
    for x in range(1, n+1):
        # As first term (x, x+d, x+2d)
        weights[x] += (n - x) // 2
        # As middle term (x-d, x, x+d)
        weights[x] += min(x - 1, n - x)
        # As last term (x-2d, x-d, x)
        weights[x] += (x - 1) // 2
        
    # Greedy construction to find a good initial solution
    # Sort by weight descending
    order = sorted(range(1, n+1), key=lambda v: -weights[v])
    
    C = set()
    # Simple greedy: add if doesn't complete a 4-AP
    # To be safe and fast, we check if adding v makes any 4-AP fully contained in C U {v}
    for v in order:
        can_add = True
        for ap_idx in four_aps_by_vertex[v]:
            ap = four_aps[ap_idx]
            # Check if other 3 are in C
            if all(u in C for u in ap if u != v):
                can_add = False
                break
        if can_add:
            C.add(v)
            
    best_weight = sum(weights[v] for v in C)
    best_set = list(C)
    
    # Branch and Bound
    # State: idx (next in order), C (set of chosen), F (set of forbidden)
    # F contains vertices u such that adding u would complete a 4-AP with elements in C.
    # i.e., exists E in 4-APs, u in E, E \ {u} subset of C.
    
    # We use a recursive function with nonlocal variables
    # To avoid recursion depth issues and overhead, we might need to be careful, 
    # but N=256 is manageable if pruning is good.
    
    # Precompute "neighbors" for faster forbidden calculation
    # For each v, and for each 4-AP E containing v, store the other 3 elements.
    # If we add v, we check for each u in E\{v}, if E\{u,v} subset of C_old, then u becomes forbidden.
    
    # Let's implement the B&B
    
    sys.setrecursionlimit(3000)
    
    def solve(idx, C, F):
        nonlocal best_weight, best_set
        
        # Upper Bound
        # Sum of weights of C + sum of weights of available vertices
        # Available: i from idx to N-1, v = order[i], if v not in F
        ub = sum(weights[v] for v in C)
        # Optimization: Precompute suffix sums of weights in the order?
        # But F changes. So we must iterate.
        # This loop is O(N). Depth N. Total O(N^2) per node? No, N nodes at depth N?
        # Tree size can be large.
        
        # We can optimize UB calculation by maintaining a running sum of available weights?
        # Or just compute it. 256 iterations is fast.
        
        for i in range(idx, n):
            v = order[i]
            if v not in F:
                ub += weights[v]
                
        if ub <= best_weight:
            return
            
        if idx == n:
            if sum(weights[v] for v in C) > best_weight:
                best_weight = sum(weights[v] for v in C)
                best_set = list(C)
            return
            
        v = order[idx]
        
        # If v is forbidden, we cannot add it. We must exclude it.
        if v in F:
            solve(idx + 1, C, F)
            return
            
        # Branch 1: Exclude v
        solve(idx + 1, C, F)
        
        # Branch 2: Include v
        # Check if v is valid (it is, since v not in F)
        C_new = C | {v}
        
        # Update Forbidden set
        # New forbidden vertices are those u such that E\{u} subset of C_new
        # Since C_new = C U {v}, the only way a vertex u becomes newly forbidden is if
        # it wasn't forbidden before, but now E\{u} subset of C U {v}.
        # This implies v is in E\{u}, so v in E.
        # So we only need to check u that are in a 4-AP with v.
        
        F_new = set(F)
        
        for ap_idx in four_aps_by_vertex[v]:
            ap = four_aps[ap_idx]
            # ap is a tuple of 4 elements including v
            for u in ap:
                if u == v:
                    continue
                if u in C_new:
                    continue # u is already chosen, not in F (F is for unchosen)
                # Check if E\{u} subset of C_new
                # E\{u} contains v.
                # Check if other 2 elements are in C (since v is in C_new)
                others = [x for x in ap if x != u and x != v]
                if all(x in C for x in others):
                    F_new.add(u)
                    
        solve(idx + 1, C_new, F_new)

    # Start B&B
    # Initial F is empty
    # Initial C is empty? 
    # Wait, the greedy solution is just a lower bound. 
    # We should start B&B from scratch to find the global optimum?
    # Or use the greedy solution to prune?
    # The B&B will explore the space. 
    # To speed up, we can sort order by weight.
    # We can also use the greedy solution's weight as initial best.
    
    # However, starting from empty C might be slow if the greedy solution is much better.
    # We can try to "fix" the greedy solution? No, B&B is exact.
    # But 2^256 is huge. Pruning is key.
    # The UB `sum(weights[v] for v in C) + sum(weights[v] for v in available)` is weak.
    
    # Let's add a timeout or node limit? No, must finish in 60s.
    # If B&B is too slow, we might time out.
    # The greedy solution is a valid solution.
    # If B&B times out, we return the best found so far.
    # But Python doesn't have a built-in timeout for functions easily without signals.
    # We can use a simple counter.
    
    node_count = 0
    MAX_NODES = 1000000 # Arbitrary limit to prevent infinite loop if pruning fails
    
    def solve_limited(idx, C, F):
        nonlocal best_weight, best_set, node_count
        
        node_count += 1
        if node_count > MAX_NODES:
            return False # Signal timeout/limit reached
            
        ub = sum(weights[v] for v in C)
        for i in range(idx, n):
            v = order[i]
            if v not in F:
                ub += weights[v]
                
        if ub <= best_weight:
            return True
            
        if idx == n:
            if sum(weights[v] for v in C) > best_weight:
                best_weight = sum(weights[v] for v in C)
                best_set = list(C)
            return True
            
        v = order[idx]
        
        if v in F:
            return solve_limited(idx + 1, C, F)
            
        # Branch 1: Exclude v
        if not solve_limited(idx + 1, C, F):
            return False
            
        # Branch 2: Include v
        C_new = C | {v}
        F_new = set(F)
        
        for ap_idx in four_aps_by_vertex[v]:
            ap = four_aps[ap_idx]
            for u in ap:
                if u == v:
                    continue
                if u in C_new:
                    continue
                others = [x for x in ap if x != u and x != v]
                if all(x in C for x in others):
                    F_new.add(u)
                    
        return solve_limited(idx + 1, C_new, F_new)

    # Run B&B
    solve_limited(0, set(), set())
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
