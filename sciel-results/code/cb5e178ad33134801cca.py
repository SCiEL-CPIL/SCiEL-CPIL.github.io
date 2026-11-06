def construct():
    from fractions import Fraction
    import time
    import math
    import random
    
    deadline = time.time() + 55  # 55 second deadline for safety (90s limit, leave margin)
    
    # Denominators are 2 to 48, indices 0 to 46
    denoms = list(range(2, 49))
    n = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # Greedy initial solution: pick largest fractions that keep sum < 1
    def greedy_solve():
        selected = []
        total = Fraction(0)
        # Sort by value descending (smallest denominator first)
        order = sorted(range(n), key=lambda i: -fracs[i])
        for i in order:
            if total + fracs[i] < 1:
                selected.append(i)
                total += fracs[i]
        return selected, total
    
    best_idxs, best_sum = greedy_solve()
    best_deficit = 1 - best_sum
    
    # Local search: try to improve by swapping or adding/removing elements
    def local_search(idx_list, total, max_iters=100000):
        nonlocal best_idxs, best_sum, best_deficit
        current_idxs = list(idx_list)
        current_total = total
        current_deficit = 1 - current_total
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            current_set = set(current_idxs)
            # Try adding any element not in set
            for i in range(n):
                if i not in current_set:
                    new_total = current_total + fracs[i]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < current_deficit:
                            current_idxs.append(i)
                            current_total = new_total
                            current_deficit = new_deficit
                            current_set.add(i)
                            improved = True
                            if current_deficit < best_deficit:
                                best_idxs = list(current_idxs)
                                best_sum = current_total
                                best_deficit = current_deficit
                            break
            if improved:
                continue
            # Try removing one element and adding another
            for i in range(n):
                if i in current_set:
                    for j in range(n):
                        if j not in current_set:
                            new_total = current_total - fracs[i] + fracs[j]
                            if 0 < new_total < 1:
                                new_deficit = 1 - new_total
                                if new_deficit < current_deficit:
                                    current_idxs = [x for x in current_idxs if x != i]
                                    current_idxs.append(j)
                                    current_total = new_total
                                    current_deficit = new_deficit
                                    current_set = set(current_idxs)
                                    improved = True
                                    if current_deficit < best_deficit:
                                        best_idxs = list(current_idxs)
                                        best_sum = current_total
                                        best_deficit = current_deficit
                                    break
                    if improved:
                        break
            # Try removing one element
            if not improved:
                for i in range(n):
                    if i in current_set:
                        new_total = current_total - fracs[i]
                        if new_total < 1 and new_total > 0:
                            new_deficit = 1 - new_total
                            # Removing increases deficit, so only useful if we can add more later
                            # Skip for now, focus on swaps
                            pass
        
        return current_idxs, current_total
    
    # Run local search from greedy
    local_search(best_idxs, best_sum)
    
    # Try multiple random starts
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        # Random subset
        random_idxs = random.sample(range(n), random.randint(5, 20))
        random_total = sum(fracs[i] for i in random_idxs)
        if random_total < 1:
            local_search(random_idxs, random_total, max_iters=50000)
    
    # Try a more systematic approach: enumerate subsets of small size
    # Since we want sum close to 1, try combinations
    # Start with the greedy solution and try to refine
    
    # Another approach: meet-in-the-middle for smaller subsets
    # Split denominators into two halves
    half = n // 2
    left_fracs = fracs[:half]
    right_fracs = fracs[half:]
    
    # Generate all subset sums for left half (up to some limit)
    left_sums = {Fraction(0): []}
    for i, f in enumerate(left_fracs):
        new_sums = {}
        for s, idxs in left_sums.items():
            new_s = s + f
            if new_s < 1:
                if new_s not in new_sums:
                    new_sums[new_s] = idxs + [i]
        # Merge
        for s, idxs in new_sums.items():
            if s not in left_sums:
                left_sums[s] = idxs
    
    right_sums = {Fraction(0): []}
    for i, f in enumerate(right_fracs):
        new_sums = {}
        for s, idxs in right_sums.items():
            new_s = s + f
            if new_s < 1:
                if new_s not in new_sums:
                    new_sums[new_s] = idxs + [half + i]
        for s, idxs in new_sums.items():
            if s not in right_sums:
                right_sums[s] = idxs
    
    # Now find best combination
    for ls, lidxs in left_sums.items():
        if time.time() > deadline - 2:
            break
        # Need rs such that ls + rs < 1, maximize ls + rs
        # So rs < 1 - ls, find largest rs < 1 - ls
        target = 1 - ls
        best_rs = None
        best_ridxs = None
        for rs, ridxs in right_sums.items():
            if rs < target:
                if best_rs is None or rs > best_rs:
                    best_rs = rs
                    best_ridxs = ridxs
        if best_rs is not None:
            total = ls + best_rs
            deficit = 1 - total
            if deficit < best_deficit:
                all_idxs = lidxs + best_ridxs
                if all_idxs:
                    best_idxs = all_idxs
                    best_sum = total
                    best_deficit = deficit
    
    if not best_idxs:
        return [0]
    
    return best_idxs
