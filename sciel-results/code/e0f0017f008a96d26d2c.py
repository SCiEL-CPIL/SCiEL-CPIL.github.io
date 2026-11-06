import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}

    def is_dissociated(S):
        if not S:
            return True
        w = sorted([weights[s] for s in S], reverse=True)
        sums = {0}
        for val in w:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    def can_add(sums, w):
        """Check if weight w can be added to existing sums without collision."""
        for s in sums:
            if s + w in sums:
                return False
        return True

    def add_weight(sums, w):
        """Add weight w to sums, returning new sums set."""
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        return sums | new_sums

    def remove_and_recompute(S):
        """Recompute all subset sums for set S."""
        sums = {0}
        for x in S:
            w = weights[x]
            sums = add_weight(sums, w)
        return sums

    best_set = []
    best_size = 0

    # Strategy 1: Multiple greedy passes with different orderings
    # Use time-aware budget: allocate ~40% of time to greedy, 60% to local search
    greedy_deadline = start_time + (deadline - start_time) * 0.4
    
    for _ in range(200):
        if time.time() > greedy_deadline:
            break
        
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        current_sums = {0}
        for x in cand_order:
            w = weights[x]
            if can_add(current_sums, w):
                current_set.append(x)
                current_sums = add_weight(current_sums, w)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

    # Strategy 2: Local Search with 1-swap, 2-swap, and add moves
    current_best = list(best_set)
    best_sums = remove_and_recompute(current_best)

    # Track recent failures to avoid cycling
    failed_moves = set()
    failed_moves_ttl = {}  # move -> time when it failed

    def try_move(move_type, *args):
        """Generic move handler with failure tracking."""
        move_key = (move_type,) + args
        now = time.time()
        if move_key in failed_moves and now - failed_moves_ttl.get(move_key, 0) < 3:
            return False
        return True

    def record_failure(move_type, *args):
        move_key = (move_type,) + args
        failed_moves.add(move_key)
        failed_moves_ttl[move_key] = time.time()

    no_improve_count = 0

    while time.time() < deadline - 2:
        if not current_best:
            break
            
        improved = False
        missing = [x for x in range(1, 23) if x not in current_best]
        
        # Try to add a random missing element
        if missing:
            for x in missing:
                if not try_move('add', x):
                    continue
                w = weights[x]
                if can_add(best_sums, w):
                    current_best.append(x)
                    best_sums = add_weight(best_sums, w)
                    improved = True
                    if len(current_best) > best_size:
                        best_size = len(current_best)
                        best_set = list(current_best)
                    break
                else:
                    record_failure('add', x)
        
        # Try 1-swap: remove one, add another
        if not improved and len(current_best) > 1 and missing:
            for x_remove in current_best:
                for x_add in missing:
                    if x_remove == x_add:
                        continue
                    if not try_move('swap1', x_remove, x_add):
                        continue
                    
                    # Compute sums without x_remove
                    sums_without = {0}
                    for x in current_best:
                        if x == x_remove:
                            continue
                        w = weights[x]
                        if can_add(sums_without, w):
                            sums_without = add_weight(sums_without, w)
                    
                    # Check if adding x_add to sums_without is dissociated
                    w_add = weights[x_add]
                    if can_add(sums_without, w_add):
                        new_sums = add_weight(sums_without, w_add)
                        current_best = [x for x in current_best if x != x_remove] + [x_add]
                        best_sums = new_sums
                        improved = True
                        break
                    else:
                        record_failure('swap1', x_remove, x_add)
                if improved:
                    break
        
        # Try 2-swap: remove two, add two (only if we're stuck)
        if not improved and len(current_best) > 2 and len(missing) >= 2:
            # Sample a few 2-swap moves
            for _ in range(15):
                if time.time() > deadline - 2:
                    break
                if len(current_best) < 2 or len(missing) < 2:
                    break
                x_remove1, x_remove2 = random.sample(current_best, 2)
                x_add1, x_add2 = random.sample(missing, 2)
                
                if not try_move('swap2', tuple(sorted([x_remove1, x_remove2])), tuple(sorted([x_add1, x_add2]))):
                    continue
                
                # Compute sums without x_remove1 and x_remove2
                sums_without = {0}
                for x in current_best:
                    if x == x_remove1 or x == x_remove2:
                        continue
                    w = weights[x]
                    if can_add(sums_without, w):
                        sums_without = add_weight(sums_without, w)
                
                # Check if adding x_add1 and x_add2 is dissociated
                w1 = weights[x_add1]
                w2 = weights[x_add2]
                if can_add(sums_without, w1):
                    sums_after1 = add_weight(sums_without, w1)
                    if can_add(sums_after1, w2):
                        new_sums = add_weight(sums_after1, w2)
                        current_best = [x for x in current_best if x != x_remove1 and x != x_remove2] + [x_add1, x_add2]
                        best_sums = new_sums
                        improved = True
                        break
                else:
                    record_failure('swap2', tuple(sorted([x_remove1, x_remove2])), tuple(sorted([x_add1, x_add2])))
        
        # If no improvement, try a "remove and re-add" strategy
        if not improved and len(current_best) > 1:
            # Remove a random element and try to rebuild
            if random.random() < 0.3:
                x_remove = random.choice(current_best)
                current_best = [x for x in current_best if x != x_remove]
                best_sums = remove_and_recompute(current_best)
                # Try to add back elements
                for x in range(1, 23):
                    if x in current_best:
                        continue
                    w = weights[x]
                    if can_add(best_sums, w):
                        current_best.append(x)
                        best_sums = add_weight(best_sums, w)
                if len(current_best) > best_size:
                    best_size = len(current_best)
                    best_set = list(current_best)
                improved = True
        
        # Track no-improve streak
        if not improved:
            no_improve_count += 1
            # Clear some failed moves periodically
            if no_improve_count % 50 == 0 and len(failed_moves) > 50:
                failed_moves.clear()
                failed_moves_ttl.clear()
        else:
            no_improve_count = 0

        # If stuck for too long, restart from best_set
        if no_improve_count > 100 and len(best_set) > len(current_best):
            current_best = list(best_set)
            best_sums = remove_and_recompute(current_best)
            no_improve_count = 0

    if not best_set:
        return [1]
        
    return best_set
