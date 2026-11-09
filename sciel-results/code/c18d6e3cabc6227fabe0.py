"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Build conflict graph: edge (a,b) if (a*b) % (a+b) == 0
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def is_valid(set_list):
        s = set(set_list)
        for a in s:
            for b in s:
                if a != b and b in conflict[a]:
                    return False
        return True
    
    def greedy_order(order):
        """Greedy: add vertices in given order if no conflict with already-added."""
        chosen = set()
        for v in order:
            if not conflict[v] & chosen:
                chosen.add(v)
        return chosen
    
    def local_search(initial_set, max_iters=5000):
        """Local search: try swaps, inserts, deletes to improve."""
        current = set(initial_set)
        best = set(current)
        best_size = len(best)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding a vertex not in current
            for v in vertices:
                if v in current:
                    continue
                if not conflict[v] & current:
                    current.add(v)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one, add one
                found_swap = False
                for rem in list(current):
                    if time.time() > deadline:
                        break
                    current.discard(rem)
                    for add in vertices:
                        if add in current:
                            continue
                        if not conflict[add] & current:
                            current.add(add)
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
                    # If no add found, put rem back
                    if rem not in current:
                        current.add(rem)
            
            if not improved:
                # Try 2-for-1: remove two, add one that's now possible
                found_2for1 = False
                rem_list = list(current)
                for i in range(len(rem_list)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(rem_list)):
                        r1, r2 = rem_list[i], rem_list[j]
                        current.discard(r1)
                        current.discard(r2)
                        for add in vertices:
                            if add in current:
                                continue
                            if not conflict[add] & current:
                                current.add(add)
                                found_2for1 = True
                                improved = True
                                break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
                if not found_2for1:
                    # Restore removed vertices
                    for r in rem_list:
                        if r not in current:
                            # Check if we can re-add
                            if not conflict[r] & current:
                                current.add(r)
                            # else leave it out (it was blocking)
            
            if len(current) > best_size:
                best = set(current)
                best_size = len(best)
            
            if not improved:
                break
        
        return best
    
    best_overall = set()
    best_size = 0
    
    # Multiple randomized restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 13)
        
        # Different ordering strategies
        if restart % 3 == 0:
            # Random order
            order = vertices[:]
            random.shuffle(order)
        elif restart % 3 == 1:
            # Sort by degree (fewest conflicts first)
            order = sorted(vertices, key=lambda v: len(conflict[v]))
            # Add some randomness
            for i in range(len(order) - 1, 0, -1):
                j = random.randint(0, i)
                order[i], order[j] = order[j], order[i]
        else:
            # Sort by degree descending (most constrained first)
            order = sorted(vertices, key=lambda v: -len(conflict[v]))
            for i in range(len(order) - 1, 0, -1):
                j = random.randint(0, i)
                order[i], order[j] = order[j], order[i]
        
        greedy_set = greedy_order(order)
        
        if len(greedy_set) > best_size:
            best_overall = set(greedy_set)
            best_size = len(greedy_set)
        
        # Local search from this starting point
        if time.time() < deadline - 2.0:
            ls_result = local_search(greedy_set, max_iters=3000)
            if len(ls_result) > best_size:
                best_overall = set(ls_result)
                best_size = len(ls_result)
    
    # Also try starting from the odd numbers baseline
    if time.time() < deadline - 5.0:
        odd_set = set(range(1, N + 1, 2))
        ls_from_odd = local_search(odd_set, max_iters=5000)
        if len(ls_from_odd) > best_size:
            best_overall = set(ls_from_odd)
            best_size = len(best_overall)
    
    # Final verification
    result = sorted(best_overall)
    if not result:
        result = [1]
    
    # Verify validity
    s = set(result)
    valid = True
    for a in s:
        for b in s:
            if a != b and b in conflict[a]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known valid set
        result = list(range(1, N + 1, 2))
    
    return result
# EVOLVE-BLOCK-END
