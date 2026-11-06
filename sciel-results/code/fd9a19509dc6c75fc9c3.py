import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    N = 256
    
    def lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute conflict graph for all pairs in [2, 256]
    # adj[i] is a set of j such that lcm(i, j) <= 256
    # We use 1-based indexing for values, so index 2 corresponds to value 2.
    # Actually, let's just use the value itself as the index for simplicity in the global graph,
    # but for the subgraph solver, we map to local indices.
    
    # Global adjacency: adj_val[v] = set of u such that lcm(u, v) <= 256
    adj_val = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= N:
                adj_val[a].add(b)
                adj_val[b].add(a)
                
    # Helper to solve MWIS on a set of values V
    def solve_mwis(values):
        if not values:
            return 0.0, []
        
        # Sort values to have a consistent order
        verts = sorted(values)
        n = len(verts)
        
        # Map value -> local index
        val_to_idx = {v: i for i, v in enumerate(verts)}
        
        # Precompute local adjacency
        # adj[i] = set of local indices j that conflict with i
        adj = [set() for _ in range(n)]
        for i in range(n):
            vi = verts[i]
            for j in range(i + 1, n):
                vj = verts[j]
                if vj in adj_val[vi]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Weights
        weights = [1.0 / v for v in verts]
        
        # Greedy initialization
        # Sort by weight descending (which is value ascending)
        # But we want to pick high weight (small value) first?
        # Greedy: pick smallest value that doesn't conflict with already picked.
        # This is equivalent to sorting by weight descending.
        
        chosen_indices = []
        available = set(range(n))
        
        # Order by weight descending (value ascending)
        order = sorted(range(n), key=lambda i: verts[i])
        
        for i in order:
            if i in available:
                chosen_indices.append(i)
                available.remove(i)
                for j in adj[i]:
                    if j in available:
                        available.remove(j)
                        
        best_sum = sum(weights[i] for i in chosen_indices)
        best_set = set(chosen_indices)
        
        # Local Search: 1-swap and 1-add
        # We keep the set of chosen indices.
        # current_set is a set of local indices.
        current_set = set(chosen_indices)
        current_sum = best_sum
        
        # Precompute conflict check for speed?
        # adj[i] is a set.
        
        # We will iterate until no improvement or time limit.
        # Use 1-swap and 2-swap moves to escape local minima.
        
        while time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try adding nodes (greedy, by weight descending)
            for i in sorted(available, key=lambda x: verts[x]):
                if i in current_set:
                    continue
                if not (adj[i] & current_set):
                    current_set.add(i)
                    current_sum += weights[i]
                    improved = True
            
            if not improved:
                # 2. Try 1-swap
                best_swap_gain = 0
                best_swap_i = -1
                best_swap_j = -1
                
                for j in available:
                    if j in current_set:
                        continue
                    C_j = adj[j] & current_set
                    if not C_j:
                        continue
                    
                    min_weight_i = min(weights[i] for i in C_j)
                    gain = weights[j] - min_weight_i
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_j = j
                        for i in C_j:
                            if weights[i] == min_weight_i:
                                best_swap_i = i
                                break
                
                if best_swap_gain > 0:
                    current_set.remove(best_swap_i)
                    current_set.add(best_swap_j)
                    current_sum += best_swap_gain
                    improved = True
            
            if not improved:
                # 3. Try 2-swap: remove i1, i2 from current_set, add j1, j2
                # j1, j2 must not conflict with each other or with current_set \ {i1, i2}
                # Gain = weights[j1] + weights[j2] - weights[i1] - weights[i2] > 0
                
                # Precompute for each j not in current_set: C_j = adj[j] & current_set
                C_map = {}
                for j in available:
                    if j not in current_set:
                        C_map[j] = adj[j] & current_set
                
                # For 2-swap, we need j1, j2 such that:
                # (C_map[j1] | C_map[j2]) is subset of {i1, i2}
                # and j1 not in adj[j2] (no conflict between j1 and j2)
                
                # To be efficient, only consider j with |C_j| <= 2
                candidates = [j for j, C in C_map.items() if 1 <= len(C) <= 2]
                
                if len(candidates) >= 2:
                    # Sort by weight descending
                    candidates.sort(key=lambda j: weights[j], reverse=True)
                    
                    best_2swap_gain = 0
                    best_2swap = None
                    
                    # Limit search to top candidates for speed
                    top_candidates = candidates[:50]
                    
                    for idx1 in range(len(top_candidates)):
                        j1 = top_candidates[idx1]
                        C1 = C_map[j1]
                        
                        for idx2 in range(idx1 + 1, len(top_candidates)):
                            j2 = top_candidates[idx2]
                            C2 = C_map[j2]
                            
                            # Check j1 and j2 don't conflict
                            if j2 in adj[j1]:
                                continue
                            
                            # Union of conflicts
                            union_C = C1 | C2
                            
                            # We need to remove exactly the nodes in union_C
                            # But we're doing 2-swap, so |union_C| must be <= 2
                            if len(union_C) > 2:
                                continue
                            
                            # If |union_C| == 0, we would have added them in step 1
                            if len(union_C) == 0:
                                continue
                            
                            # Remove nodes in union_C, add j1, j2
                            removed_nodes = list(union_C)
                            gain = weights[j1] + weights[j2] - sum(weights[i] for i in removed_nodes)
                            
                            if gain > best_2swap_gain:
                                best_2swap_gain = gain
                                best_2swap = (removed_nodes, [j1, j2])
                    
                    if best_2swap is not None and best_2swap_gain > 0:
                        removed_nodes, added_nodes = best_2swap
                        for i in removed_nodes:
                            current_set.remove(i)
                        for j in added_nodes:
                            current_set.add(j)
                        current_sum += best_2swap_gain
                        improved = True
            
            if not improved:
                break
                
        return current_sum, [verts[i] for i in current_set]

    # Strategy:
    # The problem asks for a subset of [2, 256].
    # We can decompose by "small" numbers.
    # If we include a small number x, it constrains others.
    # If we don't include any small number, we are free in [17, 256] (roughly).
    
    # Let's define "small" as 2..16.
    # Case 0: No numbers from 2..16.
    # Vertices: 17..256.
    V0 = set(range(17, 257))
    sum0, set0 = solve_mwis(V0)
    best_score = sum0
    best_candidate = set0
    
    # Case x: Include exactly one small number x?
    # Or include a set of small numbers?
    # If we include x and y (both small), they might conflict.
    # lcm(x, y) <= 256 is likely for small x, y.
    # e.g., lcm(2, 3) = 6 <= 256. Conflict.
    # lcm(2, 5) = 10 <= 256. Conflict.
    # Most small numbers conflict with each other.
    # So we likely can only pick at most 1 or 2 small numbers.
    
    # Let's try picking exactly one small number x from 2..16.
    # The rest must be from Vx = {y in [2, 256] \ {x} | lcm(x, y) > 256}.
    
    for x in range(2, 17):
        Vx = []
        for y in range(2, 257):
            if y == x:
                continue
            if lcm(x, y) > 256:
                Vx.append(y)
        
        if not Vx:
            score = 1.0 / x
            set_x = [x]
        else:
            sum_x, set_x = solve_mwis(Vx)
            score = 1.0 / x + sum_x
            set_x.append(x)
            
        if score > best_score:
            best_score = score
            best_candidate = set_x
            
        if time.time() > deadline - 5.0:
            break
            
    # What if we pick 2 small numbers?
    # e.g., x, y.
    # They must not conflict: lcm(x, y) > 256.
    # For x, y in 2..16, lcm(x, y) is small.
    # Max lcm(15, 16) = 240 <= 256.
    # So NO two numbers from 2..16 can be in the set together!
    # Because lcm(x, y) <= 256 for all x, y in 2..16.
    # Proof: Max product 15*16=240. LCM <= Product.
    # So at most one number from 2..16 can be selected.
    
    # Thus, the cases "No small" and "Exactly one small x" cover all possibilities.
    
    return best_candidate
