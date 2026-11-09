import time
import itertools
import random

_LCM = 232792560

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _try_add_element(S, sums, x, deadline):
    """Try to add element x to S if it maintains dissociation. Returns (success, new_sums)."""
    if time.time() > deadline:
        return False, sums
    
    v = _LCM // x
    # Check if adding x creates a collision
    for s in sums:
        if (s + v) in sums:
            return False, sums
    
    new_sums = sums | {s + v for s in sums}
    return True, new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    # 1. Greedy construction: Add largest numbers first
    # This tends to produce a large dissociated set because large numbers
    # have smaller weights (L/i) and are less likely to create small-integer collisions.
    S = []
    sums = {0}
    
    # Sort in descending order
    candidates = sorted(universe, reverse=True)
    
    for x in candidates:
        if time.time() > deadline:
            break
        success, new_sums = _try_add_element(S, sums, x, deadline)
        if success:
            S.append(x)
            sums = new_sums

    best_S = list(S)
    best_size = len(best_S)
    
    # 2. Local Search: Try to increase size by swapping 1 element for 2.
    # Current size K. Try S' = S - {a} + {x, y} where x, y not in S.
    # This increases size by 1.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        # Try all pairs of removal candidates from S
        # And all pairs of addition candidates from NotS
        # To limit computation, we can iterate over 'a' in S, and pairs (x,y) in NotS
        
        found_better = False
        # We want to find ANY such move that results in a dissociated set.
        # We can try them in random order to avoid bias.
        
        moves = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves.append((a, x, y))
        
        random.shuffle(moves)
        
        for a, x, y in moves:
            if time.time() > deadline - 5:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            # Check if candidate is dissociated
            # Size is best_size + 1
            if _is_dissociated(candidate):
                best_S = candidate
                best_size += 1
                improved = True
                found_better = True
                break
        
        if found_better:
            # Restart the loop with the new set
            continue
        
        # 3. If stuck, try backtracking search for larger sets
        # This is more expensive but can find larger dissociated sets
        if time.time() < deadline - 15:
            # Try to find a dissociated set of size best_size + 1 using backtracking
            target_size = best_size + 1
            if target_size <= 22:
                # Use a randomized approach: try different orderings
                for attempt in range(3):
                    if time.time() > deadline - 5:
                        break
                    
                    # Shuffle the universe and try to build a set of target_size
                    shuffled_universe = universe.copy()
                    random.shuffle(shuffled_universe)
                    
                    # Backtracking search
                    def backtrack(idx, current_set, current_sums):
                        if time.time() > deadline - 5:
                            return None
                        if len(current_set) == target_size:
                            return list(current_set)
                        if idx >= len(shuffled_universe):
                            return None
                        
                        # Pruning: not enough elements left to reach target_size
                        remaining = len(shuffled_universe) - idx
                        if remaining < target_size - len(current_set):
                            return None
                        
                        # Try adding shuffled_universe[idx]
                        x = shuffled_universe[idx]
                        success, new_sums = _try_add_element(current_set, current_sums, x, deadline)
                        if success:
                            result = backtrack(idx + 1, current_set + [x], new_sums)
                            if result is not None:
                                return result
                        
                        # Try not adding shuffled_universe[idx]
                        return backtrack(idx + 1, current_set, current_sums)
                    
                    result = backtrack(0, [], {0})
                    if result is not None and len(result) == target_size:
                        if _is_dissociated(result):
                            best_S = result
                            best_size = target_size
                            improved = True
                            break
    
    return best_S
