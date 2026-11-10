# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses greedy initialization followed by simulated annealing with time budget guard.
    """
    m = 205
    k = 2
    
    # Deadline guard: 600s limit, use 580s for safety
    deadline = time.time() + 580
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)
    
    # Precompute conflict matrix for fast lookup
    # conflict[i][j] = True if (i-j) % m or (j-i) % m is in forbidden
    # We can use a boolean array of size m x m, but that's 205x205 = 42025 entries, manageable
    # Actually, let's just use a set of forbidden pairs or compute on the fly with the set
    # For speed, let's precompute a list of conflicts for each element
    # conflicts[i] = set of j such that i and j cannot both be in the set
    conflicts = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflicts[i].add(j)
    
    # Greedy seed: try multiple orderings and keep the best
    def greedy_seed(order):
        selected = set()
        for candidate in order:
            ok = True
            for s in selected:
                if candidate in conflicts[s]:
                    ok = False
                    break
            if ok:
                selected.add(candidate)
        return selected
    
    # Try natural order and a few random orders for greedy seed
    best_seed = set()
    best_size = 0
    
    # Natural order
    seed = greedy_seed(range(m))
    if len(seed) > best_size:
        best_size = len(seed)
        best_seed = set(seed)
    
    # A few random orderings
    for _ in range(50):
        random_order = list(range(m))
        random.shuffle(random_order)
        seed = greedy_seed(random_order)
        if len(seed) > best_size:
            best_size = len(seed)
            best_seed = set(seed)
    
    # Simulated annealing
    current = best_seed.copy()
    best = best_seed.copy()
    best_size_sa = len(best)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999
    
    T = T_start
    
    # Helper: check if a set is valid (should always be true for our moves)
    def is_valid(s):
        for a in s:
            for b in s:
                if a != b and b in conflicts[a]:
                    return False
        return True
    
    # Simulated annealing loop
    while time.time() < deadline and T > T_end:
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.33:
            # Try to add a random element
            candidates_to_add = [x for x in range(m) if x not in current]
            if not candidates_to_add:
                T *= alpha
                continue
            x = random.choice(candidates_to_add)
            # Check if x conflicts with any in current
            conflict_count = 0
            for y in current:
                if x in conflicts[y]:
                    conflict_count += 1
            if conflict_count == 0:
                # No conflict, always accept
                current.add(x)
                if len(current) > best_size_sa:
                    best_size_sa = len(current)
                    best = current.copy()
            else:
                # Conflict, accept with probability exp(-conflict_count / T)
                if random.random() < np.exp(-conflict_count / T):
                    # This would create an invalid state, so we skip this move
                    # Actually, for SA on this problem, we should only consider valid states
                    # So we only accept if no conflict
                    pass
        elif move_type < 0.66:
            # Try to remove a random element
            if current:
                x = random.choice(list(current))
                # Removing is always valid
                current.remove(x)
                # We don't update best here since we're getting smaller
        else:
            # Try a swap: remove one, add another
            if current:
                x = random.choice(list(current))
                candidates_to_add = [y for y in range(m) if y not in current]
                if candidates_to_add:
                    y = random.choice(candidates_to_add)
                    # Check conflicts for y with (current - {x})
                    new_set = current - {x}
                    conflict_count = 0
                    for z in new_set:
                        if y in conflicts[z]:
                            conflict_count += 1
                    if conflict_count == 0:
                        # Valid swap
                        current.remove(x)
                        current.add(y)
                        if len(current) > best_size_sa:
                            best_size_sa = len(current)
                            best = current.copy()
        
        T *= alpha
        
        # Periodically check if we should restart from best
        if len(current) < best_size_sa - 5 and random.random() < 0.01:
            current = best.copy()
    
    # Final local improvement: try to add any remaining elements
    for x in range(m):
        if x not in best:
            ok = True
            for y in best:
                if x in conflicts[y]:
                    ok = False
                    break
            if ok:
                best.add(x)
    
    # Sort for deterministic output
    result = sorted(best)
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
