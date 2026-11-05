"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55  # Leave some buffer for overhead
    
    # 1. Precompute squares
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    def is_square(n):
        return n in squares

    # 2. Build conflict graph
    # conflicts[i] is a set of j such that i+j is a square (and i != j)
    # Also check diagonal: if 2*i is square, i cannot be in set.
    valid_nodes = []
    for i in range(1, N + 1):
        if not is_square(2 * i):
            valid_nodes.append(i)
            
    # Map value to index? No, just use values directly for simplicity in sets.
    # conflicts[val] = set of other valid vals that conflict with val
    conflicts = {v: set() for v in valid_nodes}
    val_set = set(valid_nodes)
    
    for i in range(len(valid_nodes)):
        vi = valid_nodes[i]
        for j in range(i + 1, len(valid_nodes)):
            vj = valid_nodes[j]
            if is_square(vi + vj):
                conflicts[vi].add(vj)
                conflicts[vj].add(vi)

    M = len(valid_nodes)
    if M == 0:
        return [1]

    # 3. Greedy Heuristic
    def greedy_solution(seed):
        random.seed(seed)
        candidates = set(valid_nodes)
        selected = []
        
        while candidates:
            # Calculate degree of each candidate within the candidate set
            # degree(v) = len(conflicts[v] & candidates)
            # To optimize, we can just iterate
            best_node = None
            min_deg = float('inf')
            
            # Optimization: If M is small, this is fast. M ~ 170.
            # candidates shrinks.
            for v in candidates:
                deg = len(conflicts[v] & candidates)
                if deg < min_deg:
                    min_deg = deg
                    best_node = v
                elif deg == min_deg:
                    # Random tie break
                    if random.random() < 0.5:
                        best_node = v
                        min_deg = deg
            
            if best_node is None:
                break
                
            selected.append(best_node)
            # Remove best_node and its neighbors from candidates
            candidates.remove(best_node)
            for neighbor in conflicts[best_node]:
                if neighbor in candidates:
                    candidates.remove(neighbor)
            
        return selected

    # 4. Local Search Improvement
    def local_search(selected, max_iter=1000):
        selected_set = set(selected)
        best_size = len(selected_set)
        no_improve_count = 0
        iter_count = 0
        
        while iter_count < max_iter and (time.time() - start_time) < time_limit:
            improved = False
            iter_count += 1
            
            # 1. Try to add any possible node
            # A node v can be added if v not in selected_set and conflicts[v] & selected_set is empty
            for v in valid_nodes:
                if v not in selected_set:
                    if not (conflicts[v] & selected_set):
                        selected_set.add(v)
                        improved = True
                        break # Restart loop after addition
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try Swap: Remove u, add multiple nodes
            # For each u in selected_set:
            #   temp_set = selected_set - {u}
            #   Find max independent set in (valid_nodes - temp_set) that doesn't conflict with temp_set
            #   This is equivalent to finding MIS in the subgraph induced by (valid_nodes - temp_set - neighbors(temp_set))
            #   Let's just do a greedy fill on the "available" nodes
            for u in list(selected_set):
                temp_set = selected_set - {u}
                # Nodes that can be added to temp_set
                available = set()
                for v in valid_nodes:
                    if v in temp_set:
                        continue
                    if not (conflicts[v] & temp_set):
                        available.add(v)
                
                if not available:
                    continue
                    
                # Greedy pick from available
                # We want to maximize the number of added nodes
                # Use a simple greedy: pick node with min degree in available
                new_adds = []
                avail_copy = set(available)
                while avail_copy:
                    best_v = None
                    min_deg = float('inf')
                    for v in avail_copy:
                        # Degree within avail_copy
                        deg = len(conflicts[v] & avail_copy)
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        elif deg == min_deg:
                            if random.random() < 0.5:
                                best_v = v
                                min_deg = deg
                    if best_v is None:
                        break
                    new_adds.append(best_v)
                    avail_copy.remove(best_v)
                    for nb in conflicts[best_v]:
                        if nb in avail_copy:
                            avail_copy.remove(nb)
                
                new_size = len(temp_set) + len(new_adds)
                if new_size > len(selected_set):
                    selected_set = temp_set
                    for v in new_adds:
                        selected_set.add(v)
                    improved = True
                    break # Restart after change
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0
                
        return list(selected_set)

    # 5. Main Loop: Multiple Restarts
    best_solution = []
    best_size = 0
    seed = 42
    
    # Initial greedy
    current_sol = greedy_solution(seed)
    current_sol = local_search(current_sol, max_iter=200)
    if len(current_sol) > best_size:
        best_size = len(current_sol)
        best_solution = current_sol
    
    # Randomized restarts
    while (time.time() - start_time) < time_limit:
        seed += 1
        current_sol = greedy_solution(seed)
        # Limit local search time per restart if we have many
        ls_iter = 50 if len(valid_nodes) > 100 else 200
        current_sol = local_search(current_sol, max_iter=ls_iter)
        
        if len(current_sol) > best_size:
            best_size = len(current_sol)
            best_solution = current_sol
            # Optional: Run deeper local search on new best
            # But we need to watch time.
            
    # Final verification
    def _verify(s):
        s_set = set(s)
        if not s_set:
            return False
        for a in s_set:
            if is_square(2 * a):
                return False
        for i, a in enumerate(s):
            for b in s[i+1:]:
                if is_square(a + b):
                    return False
        return True

    if _verify(best_solution):
        return sorted(best_solution)
    else:
        # Fallback
        return [1]

# EVOLVE-BLOCK-END
