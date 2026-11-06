"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    def greedy(order):
        """Greedy: include numbers in given order if it doesn't create sum 200.
        Uses bitset for speed."""
        # bitset: bit i is set if sum i is reachable. Only track 0..200.
        reachable = 1  # bit 0 set
        selected = []
        for x in order:
            # new_sums = reachable shifted left by x, masked to 0..200
            new_bits = (reachable << x) & ((1 << 201) - 1)
            if new_bits & (1 << 200):
                # Adding x would create sum 200, skip
                continue
            reachable |= new_bits
            selected.append(x)
        return selected
    
    def local_improve(selected, excluded):
        """Try to swap: remove one selected, add one or two excluded."""
        selected_set = set(selected)
        best = list(selected)
        
        # Build bitset for current selected
        reachable = 1
        for x in best:
            reachable |= (reachable << x) & ((1 << 201) - 1)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each excluded element
            for x in excluded:
                new_bits = (reachable << x) & ((1 << 201) - 1)
                if not (new_bits & (1 << 200)):
                    # Can add x directly
                    best.append(x)
                    selected_set.add(x)
                    excluded.remove(x)
                    reachable |= new_bits
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one from best, add two from excluded
            for i, y in enumerate(best):
                # Recompute without y
                reachable_no_y = 1
                for z in best:
                    if z != y:
                        reachable_no_y |= (reachable_no_y << z) & ((1 << 201) - 1)
                for x1 in excluded:
                    b1 = (reachable_no_y << x1) & ((1 << 201) - 1)
                    if b1 & (1 << 200):
                        continue
                    for x2 in excluded:
                        if x2 == x1:
                            continue
                        b2 = (b1 << x2) & ((1 << 201) - 1)
                        if b2 & (1 << 200):
                            continue
                        # Swap y for x1, x2
                        best.remove(y)
                        selected_set.discard(y)
                        best.extend([x1, x2])
                        selected_set.update([x1, x2])
                        excluded.discard(x1)
                        excluded.discard(x2)
                        reachable = 1
                        for z in best:
                            reachable |= (reachable << z) & ((1 << 201) - 1)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
        return best
    
    # Try multiple orderings
    all_numbers = list(range(1, 101))
    best_result = []
    
    # Descending order
    r = greedy(all_numbers[::-1])
    if len(r) > len(best_result):
        best_result = r
    
    # Ascending order
    r = greedy(all_numbers)
    if len(r) > len(best_result):
        best_result = r
    
    # Random shuffles
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 10:
        order = all_numbers[:]
        rng.shuffle(order)
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r
            attempts = 0
        else:
            attempts += 1
        if attempts > 50:
            break
        attempts += 1
    
    # Local improvement
    selected_set = set(best_result)
    excluded = [x for x in all_numbers if x not in selected_set]
    best_result = local_improve(best_result, excluded)
    
    if not best_result:
        return [100]
    
    return best_result
# EVOLVE-BLOCK-END
