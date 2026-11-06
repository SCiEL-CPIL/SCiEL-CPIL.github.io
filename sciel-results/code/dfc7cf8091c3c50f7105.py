"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    
    Strategy:
    1. Try multiple greedy orderings with bitset-based reachability tracking
    2. Local search with efficient swap evaluation
    3. Deadline guard: return best found before 85s wall-clock limit
    """
    deadline = time.time() + 85.0  # safety margin
    TARGET = 200
    N = 100
    
    def compute_reachable_bitset(elements):
        """Compute subset sums up to TARGET using bitset operations."""
        reachable = 1  # bit 0 is set, representing sum 0
        for x in elements:
            if x <= TARGET:
                reachable |= (reachable << x)
        # Mask to only keep bits up to TARGET
        mask = (1 << (TARGET + 1)) - 1
        return reachable & mask
    
    def can_add(bitset, x):
        """Check if adding x would create a subset sum of TARGET."""
        # If (TARGET - x) is reachable, then adding x creates sum TARGET
        if x > TARGET:
            return True  # Always safe, can't contribute to sum TARGET
        return not (bitset & (1 << (TARGET - x)))
    
    def greedy_order(order, target=TARGET):
        """Run greedy with given element order, return selected set and bitset."""
        selected = set()
        reachable = 1  # bitset with bit 0 set
        for x in order:
            if can_add(reachable, x):
                selected.add(x)
                if x <= target:
                    reachable |= (reachable << x)
                    mask = (1 << (target + 1)) - 1
                    reachable &= mask
        return selected, reachable
    
    # --- Phase 1: Try multiple greedy orderings ---
    best_set = set()
    best_size = 0
    best_reachable = 1
    
    orders = []
    # Forward: 1 to 100
    orders.append(list(range(1, 101)))
    # Backward: 100 to 1
    orders.append(list(range(100, 0, -1)))
    # Random orders (a few)
    random.seed(42)
    for _ in range(20):
        order = list(range(1, 101))
        random.shuffle(order)
        orders.append(order)
    # Large-first with small fill: 100,99,98,...,50, then 1,2,3,...,49
    orders.append(list(range(100, 49, -1)) + list(range(1, 50)))
    # Small-first with large fill: 1,2,3,...,50, then 100,99,98,...,51
    orders.append(list(range(1, 51)) + list(range(100, 50, -1)))
    
    for order in orders:
        if time.time() > deadline:
            break
        selected, reachable = greedy_order(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_reachable = reachable
    
    # --- Phase 2: Local search with efficient swap evaluation ---
    # For each element in best_set, try removing it and adding an excluded element
    # We need to efficiently check if a swap is valid
    
    excluded = set(range(1, 101)) - best_set
    
    improved = True
    max_iterations = 100
    iteration = 0
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try single swaps: remove one element, add one excluded element
        for to_remove in list(best_set):
            if time.time() > deadline:
                break
            
            # Compute reachable without to_remove
            # This is expensive, so we do it carefully
            candidate_elements = [x for x in best_set if x != to_remove]
            reach_without = compute_reachable_bitset(candidate_elements)
            
            # Try adding each excluded element
            for to_add in sorted(excluded, reverse=True):
                if to_add in best_set:
                    continue
                if can_add(reach_without, to_add):
                    # Check if this improves cardinality (it won't for single swap,
                    # but we might enable future improvements)
                    # For now, just track if we can make a neutral swap that
                    # opens up more possibilities
                    # Actually, for cardinality maximization, we need to find
                    # a way to add an element without removing one, or find a
                    # 2-for-1 swap
                    pass
        
        # Try 2-for-1 swaps: remove two elements, add one excluded element
        # This can increase cardinality if we remove 2 and add 3, but that's complex
        # For now, let's try a different approach: try to add an excluded element
        # by removing a minimal set of elements
        
        # Alternative: try different greedy orderings with the current best as a seed
        # Modify the order to prioritize adding excluded elements early
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Create an order that tries to include excluded elements
            order = []
            # First, try to add excluded elements (in random order)
            excl_list = list(excluded)
            random.shuffle(excl_list)
            order.extend(excl_list)
            # Then add the rest
            remaining = [x for x in range(1, 101) if x not in excl_list]
            random.shuffle(remaining)
            order.extend(remaining)
            
            selected, reachable = greedy_order(order)
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
                best_reachable = reachable
                excluded = set(range(1, 101)) - best_set
                improved = True
                break
    
    # --- Phase 3: Try to add elements by removing minimal subsets ---
    # For each excluded element, find if we can add it by removing some elements
    # This is a hitting set problem, but we can try small removals
    
    for to_add in sorted(excluded, reverse=True):
        if time.time() > deadline:
            break
        
        # We need to find a subset R of best_set such that:
        # 1. No subset of (best_set - R) + {to_add} sums to 200
        # 2. |R| is minimized (ideally 0, but that's impossible since to_add is excluded)
        
        # Try removing 1 element
        for to_remove in best_set:
            candidate = (best_set - {to_remove}) | {to_add}
            # Check if any subset sums to 200
            reach = compute_reachable_bitset(list(candidate))
            if not (reach & (1 << TARGET)):
                # Valid! And same cardinality, but might enable future improvements
                # For now, accept if it doesn't reduce cardinality
                if len(candidate) >= best_size:
                    best_set = candidate
                    best_size = len(best_set)
                    excluded = set(range(1, 101)) - best_set
                    break
        
        # Try removing 2 elements
        if len(excluded) > 0 and time.time() < deadline:
            best_set_list = list(best_set)
            for i in range(len(best_set_list)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_set_list)):
                    candidate = (best_set - {best_set_list[i], best_set_list[j]}) | {to_add}
                    reach = compute_reachable_bitset(list(candidate))
                    if not (reach & (1 << TARGET)):
                        if len(candidate) >= best_size:
                            best_set = candidate
                            best_size = len(best_set)
                            excluded = set(range(1, 101)) - best_set
                            break
                if len(excluded) < len(set(range(1, 101)) - best_set):
                    break
    
    return list(best_set)
# EVOLVE-BLOCK-END
