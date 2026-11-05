def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = 85.0  # safety margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute conflicts: conflict[a][b] = True if lcm(a,b) <= 256
    # We'll work with numbers 2..256
    N = 256
    nums = list(range(2, N + 1))
    idx = {x: i for i, x in enumerate(nums)}
    n = len(nums)
    
    # Build conflict set
    # conflict_mask[i] = set of j such that lcm(nums[i], nums[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            if a * b // gcd(a, b) <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy: sort by 1/x descending (i.e., x ascending), add if no conflict with selected
    # But this is suboptimal. Let's do a better greedy + local search.
    
    def reciprocal_sum(selected_set):
        return sum(1.0 / nums[i] for i in selected_set)
    
    # Initial solution: greedy by ascending x (highest 1/x first)
    selected = set()
    for i in range(n):
        if time.time() > deadline:
            break
        if all(j not in selected for j in conflict[i]):
            selected.add(i)
    
    best_sum = reciprocal_sum(selected)
    best_selected = set(selected)
    
    # Local search: try to swap or add/remove
    # Try adding unselected elements
    def try_improve(selected_set, budget):
        improved = True
        while improved and time.time() < budget:
            improved = False
            # Try adding each unselected element
            for i in range(n):
                if i not in selected_set:
                    if all(j not in selected_set for j in conflict[i]):
                        new_set = selected_set | {i}
                        if reciprocal_sum(new_set) > best_sum:
                            selected_set = new_set
                            best_sum = reciprocal_sum(selected_set)
                            improved = True
            # Try removing one and adding two (or one better)
            # Simpler: try removing each selected and see if we can add better
            for i in list(selected_set):
                # Remove i, try to add elements that were blocked by i
                temp_set = selected_set - {i}
                best_addition = None
                best_gain = 0
                for j in range(n):
                    if j not in temp_set and j != i:
                        if all(k not in temp_set for k in conflict[j]):
                            gain = 1.0 / nums[j]
                            if gain > best_gain:
                                best_gain = gain
                                best_addition = j
                # Check if removing i and adding best_addition improves
                if best_addition is not None:
                    new_set = temp_set | {best_addition}
                    new_sum = reciprocal_sum(new_set)
                    if new_sum > best_sum + 1e-12:
                        selected_set = new_set
                        best_sum = new_sum
                        improved = True
                # Also just removing i might not help, but check
                else:
                    new_sum = reciprocal_sum(temp_set)
                    if new_sum > best_sum + 1e-12:
                        selected_set = temp_set
                        best_sum = new_sum
                        improved = True
        return selected_set, best_sum
    
    selected, best_sum = try_improve(selected, deadline - 5)
    best_selected = set(selected)
    
    # Final: try random restarts with different orderings if time permits
    import random
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 2 and attempts < 50:
        attempts += 1
        # Random greedy
        order = list(range(n))
        random.shuffle(order)
        # Sort by 1/x with some randomness
        order.sort(key=lambda i: (nums[i] + random.random() * 10))
        sel = set()
        for i in order:
            if all(j not in sel for j in conflict[i]):
                sel.add(i)
        s = reciprocal_sum(sel)
        if s > best_sum:
            best_sum = s
            best_selected = set(sel)
            # Local improve
            sel, s = try_improve(sel, deadline - 2)
            if s > best_sum:
                best_sum = s
                best_selected = set(sel)
    
    # Return as sorted list of actual values
    result = sorted(nums[i] for i in best_selected)
    return result
