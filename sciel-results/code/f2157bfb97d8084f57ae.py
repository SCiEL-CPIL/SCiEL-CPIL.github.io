"""Rewritten program for large interval sets avoiding a prescribed subset sum."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    
    Strategy:
    1. Greedy construction with multiple orderings using bitset reachability
    2. Local search: try to add excluded elements by removing minimal conflicting subsets
    3. Deadline guard for safety
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill
    TARGET = 200
    N = 100
    ALL_ELEMENTS = set(range(1, N + 1))
    
    def compute_reachable_bitset(elements):
        """Compute subset sums up to TARGET using bitset operations."""
        reachable = 1  # bit 0 is set, representing sum 0
        for x in elements:
            if x <= TARGET:
                reachable |= (reachable << x)
        mask = (1 << (TARGET + 1)) - 1
        return reachable & mask
    
    def can_add(bitset, x):
        """Check if adding x would create a subset sum of TARGET."""
        if x > TARGET:
            return True
        return not (bitset & (1 << (TARGET - x)))
    
    def greedy_order(order, target=TARGET):
        """Run greedy with given element order, return selected set and bitset."""
        selected = set()
        reachable = 1
        for x in order:
            if can_add(reachable, x):
                selected.add(x)
                if x <= target:
                    reachable |= (reachable << x)
                    mask = (1 << (target + 1)) - 1
                    reachable &= mask
        return selected, reachable
    
    def is_valid(elements):
        """Check if no subset of elements sums to TARGET."""
        reach = compute_reachable_bitset(elements)
        return not (reach & (1 << TARGET))
    
    # --- Phase 1: Try multiple greedy orderings ---
    best_set = set()
    best_size = 0
    best_reachable = 1
    
    orders = []
    # Forward: 1 to 100
    orders.append(list(range(1, 101)))
    # Backward: 100 to 1
    orders.append(list(range(100, 0, -1)))
    # Large-first with small fill
    orders.append(list(range(100, 49, -1)) + list(range(1, 50)))
    # Small-first with large fill
    orders.append(list(range(1, 51)) + list(range(100, 50, -1)))
    # Even then odd
    orders.append(list(range(2, 101, 2)) + list(range(1, 101, 2)))
    # Odd then even
    orders.append(list(range(1, 101, 2)) + list(range(2, 101, 2)))
    # Random orders
    random.seed(42)
    for _ in range(50):
        order = list(range(1, 101))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        selected, reachable = greedy_order(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_reachable = reachable
    
    # --- Phase 2: Local search to add excluded elements ---
    # For each excluded element, try to add it by removing a minimal set of conflicting elements
    improved = True
    max_rounds = 50
    round_num = 0
    
    while improved and time.time() < deadline and round_num < max_rounds:
        improved = False
        round_num += 1
        
        excluded = ALL_ELEMENTS - best_set
        if not excluded:
            break
        
        # Sort excluded elements by value (larger first tends to be more "expensive" in terms of conflicts)
        for to_add in sorted(excluded, reverse=True):
            if time.time() > deadline:
                break
            if to_add in best_set:
                continue
            
            # Try adding to_add with removal of 0, 1, 2, or 3 elements
            found_better = False
            
            # Try removal of 0 elements (just add)
            if can_add(best_reachable, to_add):
                best_set.add(to_add)
                if to_add <= TARGET:
                    best_reachable |= (best_reachable << to_add)
                    mask = (1 << (TARGET + 1)) - 1
                    best_reachable &= mask
                best_size += 1
                improved = True
                found_better = True
                continue
            
            # Try removal of 1 element
            for to_remove in list(best_set):
                if time.time() > deadline:
                    break
                candidate = (best_set - {to_remove}) | {to_add}
                if len(candidate) <= best_size:
                    continue  # We want to increase or maintain size, but since we're adding 1 and removing 1, size is same
                    # Actually for improvement we need net gain, so skip same-size swaps unless they enable future gains
                    # Let's allow same-size swaps that might enable future improvements
                reach = compute_reachable_bitset(list(candidate))
                if not (reach & (1 << TARGET)):
                    best_set = candidate
                    best_size = len(best_set)
                    best_reachable = reach
                    improved = True
                    found_better = True
                    break
            
            if found_better:
                continue
            
            # Try removal of 2 elements
            if time.time() > deadline:
                break
            best_list = list(best_set)
            for i in range(len(best_list)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_list)):
                    if time.time() > deadline:
                        break
                    candidate = (best_set - {best_list[i], best_list[j]}) | {to_add}
                    if len(candidate) < best_size:
                        continue
                    reach = compute_reachable_bitset(list(candidate))
                    if not (reach & (1 << TARGET)):
                        best_set = candidate
                        best_size = len(best_set)
                        best_reachable = reach
                        improved = True
                        found_better = True
                        break
                if found_better:
                    break
            
            if found_better:
                continue
            
            # Try removal of 3 elements
            if time.time() > deadline:
                break
            best_list = list(best_set)
            n = len(best_list)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            break
                        candidate = (best_set - {best_list[i], best_list[j], best_list[k]}) | {to_add}
                        if len(candidate) < best_size:
                            continue
                        reach = compute_reachable_bitset(list(candidate))
                        if not (reach & (1 << TARGET)):
                            best_set = candidate
                            best_size = len(best_set)
                            best_reachable = reach
                            improved = True
                            found_better = True
                            break
                    if found_better:
                        break
                if found_better:
                    break
    
    # --- Phase 3: Final greedy attempts with different seeds ---
    random.seed(123)
    for _ in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        selected, reachable = greedy_order(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_reachable = reachable
    
    # Ensure nonempty
    if not best_set:
        return [1]
    
    return list(best_set)
# EVOLVE-BLOCK-END
