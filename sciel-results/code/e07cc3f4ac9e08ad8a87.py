"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def can_add(x, reachable):
        """Check if adding x to current reachable sums creates 200."""
        for s in reachable:
            if s + x == 200:
                return False
        return True
    
    def greedy_order(order):
        """Greedily select numbers in given order, avoiding subset sum 200."""
        selected = []
        reachable = {0}
        for x in order:
            if can_add(x, reachable):
                selected.append(x)
                new_reachable = set()
                for s in reachable:
                    ns = s + x
                    if ns <= 200:
                        new_reachable.add(ns)
                reachable |= new_reachable
        return selected
    
    # Try multiple orderings and pick the best
    best_result = []
    
    # Ordering 1: descending
    result1 = greedy_order(range(100, 0, -1))
    best_result = result1
    
    # Ordering 2: ascending
    result2 = greedy_order(range(1, 101))
    if len(result2) > len(best_result):
        best_result = result2
    
    # Ordering 3: large first, then small
    order3 = list(range(100, 50, -1)) + list(range(1, 51))
    result3 = greedy_order(order3)
    if len(result3) > len(best_result):
        best_result = result3
    
    # Ordering 4: alternate large and small
    order4 = []
    for i in range(50):
        order4.append(100 - i)
        order4.append(i + 1)
    result4 = greedy_order(order4)
    if len(result4) > len(best_result):
        best_result = result4
    
    # Verify the result
    def has_subset_sum_200(nums):
        reachable = {0}
        for x in nums:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 in new:
                return True
            reachable |= new
        return 200 in reachable
    
    if has_subset_sum_200(best_result):
        # Should not happen, but fallback to descending
        return greedy_order(range(100, 0, -1))
    
    return best_result
# EVOLVE-BLOCK-END
