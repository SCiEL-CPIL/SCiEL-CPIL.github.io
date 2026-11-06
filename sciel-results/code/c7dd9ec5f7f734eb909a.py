import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill (using 55s for safety)
    
    n = 256
    max_sum = 2 * n
    
    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Step 1: Filter candidates based on diagonal constraint (2*a not square)
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    
    # Build adjacency list for the conflict graph
    # conflict[a] is a set of b such that a+b is a square
    conflict = {a: set() for a in candidates}
    
    # Efficiently build conflicts
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
                
    def greedy_min_degree(pool):
        """
        Greedy algorithm: repeatedly pick the node with minimum degree in the remaining pool,
        add to solution, remove it and its neighbors from pool.
        """
        pool = set(pool)
        selected = []
        while pool:
            min_deg_node = None
            min_deg = float('inf')
            for node in pool:
                # Count conflicts with other nodes in the pool
                deg = len(conflict[node] & pool)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = node
                    if min_deg == 0:
                        break
                        
            if min_deg_node is None:
                break
                
            selected.append(min_deg_node)
            # Remove node and its neighbors from pool
            pool.discard(min_deg_node)
            pool.difference_update(conflict[min_deg_node])
            
        return selected

    def greedy_randomized(pool, seed=None):
        """
        Randomized greedy: pick random node from nodes with minimum degree.
        """
        if seed is not None:
            random.seed(seed)
        pool = set(pool)
        selected = []
        while pool:
            degrees = {node: len(conflict[node] & pool) for node in pool}
            min_deg = min(degrees.values())
            candidates_for_pick = [node for node, deg in degrees.items() if deg == min_deg]
            node = random.choice(candidates_for_pick)
            
            selected.append(node)
            pool.discard(node)
            pool.difference_update(conflict[node])
            
        return selected

    def local_search(s, time_limit):
        """
        Local search with:
        1. Add any valid vertex.
        2. 1-for-2 swap (remove 1, add 2 non-conflicting).
        3. Perturbation (remove k random, refill greedily).
        """
        current = set(s)
        in_set = set(current)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex that is not in set and has no conflicts with set
            for v in candidates:
                if v in in_set:
                    continue
                # Check if v has any neighbor in current
                if not (conflict[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    break # Restart loop to check new state
            
            if not improved:
                # 2. Try 1-for-2 swap
                # For each u in current, find candidates v not in current such that 
                # conflicts[v] & current == {u}
                # If we find two such v1, v2 that are not adjacent, swap u -> {v1, v2}
                
                swap_done = False
                for u in list(current):
                    candidates_for_swap = []
                    for v in candidates:
                        if v in in_set:
                            continue
                        # v is blocked by current if (conflict[v] & in_set) is non-empty
                        # We want (conflict[v] & in_set) to be exactly {u}
                        if u not in conflict[v]:
                            continue
                        
                        common = conflict[v] & in_set
                        if len(common) == 1 and u in common:
                            candidates_for_swap.append(v)
                    
                    if len(candidates_for_swap) >= 2:
                        # Find two non-adjacent candidates
                        found_swap = False
                        for i in range(len(candidates_for_swap)):
                            if found_swap: break
                            for j in range(i + 1, len(candidates_for_swap)):
                                if candidates_for_swap[i] not in conflict[candidates_for_swap[j]]:
                                    # Swap!
                                    current.remove(u)
                                    in_set.remove(u)
                                    current.add(candidates_for_swap[i])
                                    in_set.add(candidates_for_swap[i])
                                    current.add(candidates_for_swap[j])
                                    in_set.add(candidates_for_swap[j])
                                    improved = True
                                    swap_done = True
                                    found_swap = True
                                    break
                        if swap_done:
                            break # Restart outer loop
                
                if not swap_done:
                    # 3. Perturbation: Remove 1-3 random vertices and refill
                    # This helps escape local minima
                    if time.time() > time_limit - 2:
                        break
                        
                    # Random removal
                    k_remove = random.randint(1, 3)
                    to_remove = random.sample(list(current), min(k_remove, len(current)))
                    for v in to_remove:
                        current.remove(v)
                        in_set.remove(v)
                    
                    # Refill greedily
                    refill_improved = True
                    while refill_improved:
                        refill_improved = False
                        for v in candidates:
                            if v in in_set:
                                continue
                            if not (conflict[v] & in_set):
                                current.add(v)
                                in_set.add(v)
                                refill_improved = True
                                break
                                
                    # If size increased, we are improved. If not, we still accept 
                    # the perturbation to allow further search? 
                    # Usually, we accept if size >= previous best, or with probability.
                    # For simplicity, if size didn't increase, we might have lost ground.
                    # But perturbation is meant to jump out. 
                    # Let's only mark improved if size increased or stayed same after perturbation 
                    # and we found a new state? No, standard local search accepts if better.
                    # Here, if we didn't find an immediate improvement (add or swap), 
                    # and perturbation didn't immediately lead to a larger set, 
                    # we should probably stop or accept the perturbed state if it's not worse?
                    # Actually, the "improved" flag controls the while loop.
                    # If we did a perturbation and the size is now smaller, we haven't "improved".
                    # But we want to continue searching from this new state.
                    # So we should set improved = True to continue the loop, 
                    # but we need to be careful not to loop forever if no improvement is ever found.
                    # However, the perturbation changes the state, so the next iteration might find an improvement.
                    # To prevent infinite loops without progress, we can limit perturbation attempts 
                    # or accept that we are moving through the state space.
                    # Let's set improved = True to continue, but add a counter for no-improvement steps?
                    # For now, let's just continue. The time limit will stop it.
                    improved = True
                    
        return current

    best_solution = []
    
    # 1. Deterministic Min-Degree Greedy
    sol = greedy_min_degree(candidates)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # 2. Randomized Restart Greedy + Local Search
    # We will run multiple restarts. For each, we do a greedy init and then local search.
    # We keep the best solution found so far.
    
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() - start_time > deadline - 5.0:
            break
            
        # Greedy init
        sol = greedy_randomized(candidates, seed=i)
        
        # Local search with a time limit for this restart
        # Allocate remaining time minus a buffer for final checks
        remaining_time = deadline - time.time()
        # Give each local search a portion of the remaining time
        # Or just run until the global deadline is close
        # Let's run local search with a limit of 2 seconds per restart? 
        # Or dynamically: if we have > 5s left, give it 2s. If < 5s, give it remaining - 1s.
        ls_limit = time.time() + min(2.0, max(0.1, remaining_time - 1.0))
        
        sol_ls = local_search(sol, ls_limit)
        
        if len(sol_ls) > len(best_solution):
            best_solution = sol_ls
            
    # 3. Final Polish: Try to add any remaining candidate
    best_set = set(best_solution)
    improved = list(best_solution)
    for a in candidates:
        if a not in best_set:
            # Check if a conflicts with any in best_set
            if not (conflict[a] & best_set):
                improved.append(a)
                best_set.add(a)
                
    if len(improved) > len(best_solution):
        best_solution = improved
        
    # One last quick local search on the best solution if time permits
    if time.time() < deadline - 1.0:
        final_ls = local_search(best_solution, deadline - 0.5)
        if len(final_ls) > len(best_solution):
            best_solution = final_ls

    return best_solution
