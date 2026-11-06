"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # First, filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    # Build conflict graph: which valid numbers conflict with which
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # For each valid number, find all other valid numbers it conflicts with
    # a and b conflict if a + b is a perfect square (a != b)
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])
    
    # Greedy: pick elements with minimum degree first (least conflicts)
    # This is a heuristic for maximum independent set
    
    def greedy_min_degree():
        if time.time() > deadline:
            return None
        
        remaining = set(range(num_valid))
        selected = []
        
        while remaining:
            # Find element with minimum degree among remaining
            min_deg = float('inf')
            best_idx = -1
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.discard(best_idx)
            # Remove all neighbors of best_idx from remaining
            for neighbor in conflict[best_idx]:
                remaining.discard(neighbor)
        
        return [valid_numbers[i] for i in selected]
    
    # Try greedy min-degree
    best_result = greedy_min_degree()
    
    if best_result is None:
        # Fallback to simple valid set
        return list(range(1, 257, 3))
    
    best_size = len(best_result)
    
    # Try local search: remove one element, try to add another
    def local_search(initial_set):
        if time.time() > deadline:
            return initial_set
        
        current = set(initial_set)
        current_size = len(current)
        best = list(current)
        
        # Precompute conflicts for quick lookup
        in_set = set(current)
        
        # Try removing each element and seeing if we can add more
        for to_remove in list(current):
            if time.time() > deadline:
                break
            
            trial_set = current - {to_remove}
            # Try adding each number not in trial_set
            for candidate in range(1, n + 1):
                if candidate in trial_set:
                    continue
                if 2 * candidate in squares:
                    continue
                # Check if candidate conflicts with any in trial_set
                can_add = True
                for s in squares:
                    other = s - candidate
                    if other in trial_set and other != candidate:
                        can_add = False
                        break
                    if other == candidate and 2 * candidate in squares:
                        can_add = False
                        break
                if can_add:
                    new_size = len(trial_set) + 1
                    if new_size > current_size:
                        current = trial_set | {candidate}
                        current_size = new_size
                        best = list(current)
                        break
        
        return best
    
    improved = local_search(best_result)
    if len(improved) > best_size:
        best_result = improved
        best_size = len(improved)
    
    # Return sorted result
    return sorted(best_result)
# EVOLVE-BLOCK-END
