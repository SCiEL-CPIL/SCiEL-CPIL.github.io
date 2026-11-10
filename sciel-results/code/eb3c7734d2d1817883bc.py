# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses bitset-optimized greedy construction and local search (1-move, 1-swap, 2-swap).
    """
    m = 205
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute conflict masks for each element
    # conflict_mask[i] is a bitmask where bit j is set if (i-j)%m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            mask |= (1 << j)
        conflict_mask[i] = mask
    
    def greedy_construct(order):
        """Greedy construction using bitsets."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            # Check if candidate conflicts with any selected element
            if (selected_mask & conflict_mask[candidate]) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def local_search_1move(selected_list, selected_mask):
        """Try to add any valid element. Repeat until no more can be added."""
        selected_list = list(selected_list)
        selected_mask = selected_mask
        improved = True
        while improved:
            improved = False
            for a in range(m):
                if (selected_mask & (1 << a)) == 0: # Not selected
                    if (selected_mask & conflict_mask[a]) == 0:
                        selected_mask |= (1 << a)
                        selected_list.append(a)
                        improved = True
        return selected_list, selected_mask

    def local_search_1swap(selected_list, selected_mask):
        """
        Try to swap one element out and one in. 
        This maintains size but explores the space. 
        We accept a swap if it allows a subsequent 1-move to increase size, 
        or simply accept valid swaps to shuffle the set.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        improved = True
        # Limit iterations to prevent infinite loops of swapping back and forth
        # We track visited masks? No, too expensive. Just limit steps.
        steps = 0
        max_steps = 5000
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            # Try removing each selected element
            for r in list(selected_list):
                # Temporarily remove r
                temp_mask = selected_mask & ~(1 << r)
                
                # Try to add any non-selected element
                for a in range(m):
                    if (temp_mask & (1 << a)) == 0: # a is not in temp_mask
                        if (temp_mask & conflict_mask[a]) == 0:
                            # Valid swap: remove r, add a
                            # Update state
                            selected_mask = temp_mask | (1 << a)
                            # Update list: replace r with a
                            idx = selected_list.index(r)
                            selected_list[idx] = a
                            improved = True
                            break
                if improved:
                    break
        return selected_list, selected_mask

    def local_search_2swap(selected_list, selected_mask):
        """
        Try to remove two elements and add two.
        This can increase size by 0, but helps escape local optima.
        We look for a swap that results in a state where a 1-move can be applied.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        improved = True
        steps = 0
        max_steps = 1000 # 2-swap is expensive
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            # Sample random pairs to remove to keep it fast
            # Or iterate? 205 choose 2 is ~20k pairs. 
            # Let's iterate over a subset or random samples if time permits.
            # For now, random samples to keep average case fast.
            
            # Try a few random 2-swaps
            for _ in range(100):
                if len(selected_list) < 2:
                    break
                # Pick two distinct elements to remove
                r1, r2 = random.sample(selected_list, 2)
                temp_mask = selected_mask & ~(1 << r1) & ~(1 << r2)
                
                # Try to add two elements
                # We need to find a1, a1 != a2, not in temp_mask, valid with temp_mask and each other
                # Simple heuristic: try to add a1, then check a2
                added_any = False
                for a1 in range(m):
                    if (temp_mask & (1 << a1)) == 0:
                        if (temp_mask & conflict_mask[a1]) == 0:
                            temp_mask_with_a1 = temp_mask | (1 << a1)
                            for a2 in range(m):
                                if (temp_mask_with_a1 & (1 << a2)) == 0:
                                    if (temp_mask_with_a1 & conflict_mask[a2]) == 0:
                                        # Valid 2-swap: remove r1,r2 add a1,a2
                                        # Update
                                        selected_mask = temp_mask_with_a1 | (1 << a2)
                                        # Update list
                                        # Remove r1, r2
                                        selected_list.remove(r1)
                                        selected_list.remove(r2)
                                        selected_list.append(a1)
                                        selected_list.append(a2)
                                        improved = True
                                        added_any = True
                                        break
                                if added_any:
                                    break
                    if added_any:
                        break
                if improved:
                    break
        return selected_list, selected_mask

    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_solution = None
    best_size = 0
    best_mask = 0
    
    # 1. Greedy with multiple orders
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
    ]
    
    # Generate degree-based order
    # Degree of i is number of j such that (i-j)%m in forbidden
    # This is just len(forbidden) for all i? No, the graph is regular if forbidden is symmetric?
    # x^2 mod m. If d is forbidden, m-d is also forbidden? 
    # (m-x)^2 = m^2 - 2mx + x^2 = x^2 mod m. Yes, symmetric.
    # So degree is constant for all nodes? 
    # Wait, the set of differences is fixed. The graph is a circulant graph.
    # So all nodes have the same degree.
    # However, random orders are useful.
    
    rng = random.Random(42)
    for _ in range(20):
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        sol, mask = greedy_construct(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol[:]
            best_mask = mask
    
    # 2. Local Search on best solution
    if best_solution is not None:
        best_solution, best_mask = local_search_1move(best_solution, best_mask)
        if len(best_solution) > best_size:
            best_size = len(best_solution)
        
        # Try 1-swap to shuffle, then 1-move again
        sol_swap, mask_swap = local_search_1swap(best_solution, best_mask)
        sol_swap, mask_swap = local_search_1move(sol_swap, mask_swap)
        if len(sol_swap) > best_size:
            best_size = len(sol_swap)
            best_solution = sol_swap
            best_mask = mask_swap
            
        # Try 2-swap
        sol_2swap, mask_2swap = local_search_2swap(best_solution, best_mask)
        sol_2swap, mask_2swap = local_search_1move(sol_2swap, mask_2swap)
        if len(sol_2swap) > best_size:
            best_size = len(sol_2swap)
            best_solution = sol_2swap
            best_mask = mask_2swap

    # 3. Random Restarts with Local Search
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        order = list(range(m))
        rng.shuffle(order)
        sol, mask = greedy_construct(order)
        sol, mask = local_search_1move(sol, mask)
        
        # Probabilistically try 1-swap
        if len(sol) < best_size + 2: # Only if we might improve
             sol, mask = local_search_1swap(sol, mask)
             sol, mask = local_search_1move(sol, mask)
             
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
            best_mask = mask
            
        # Check time
        if time.time() > deadline - 10:
            break

    # Final polish
    if best_solution is not None and time.time() < deadline - 2:
        final_sol, final_mask = local_search_1move(best_solution, best_mask)
        if len(final_sol) > best_size:
            best_size = len(final_sol)
            best_solution = final_sol

    if best_solution is None:
        best_solution = [0]
    
    return np.array(best_solution, dtype=np.int64)

# EVOLVE-BLOCK-END
