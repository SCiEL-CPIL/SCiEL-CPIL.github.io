import random
import time


def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    def try_order(order):
        S = set()
        sums = {0}
        for x in order:
            if (200 - x) not in sums:
                S.add(x)
                new_sums = set()
                for s in sums:
                    ns = s + x
                    if ns <= 200:
                        new_sums.add(ns)
                sums.update(new_sums)
        return S

    # Compute degree for each vertex: number of subsets of [1,100]\{x} that sum to 200-x
    # A vertex x has "degree" = number of ways to form 200-x from other elements
    # Higher degree means more constraints, so we prefer lower-degree vertices first
    def compute_degrees():
        degrees = {}
        for x in range(1, 101):
            target = 200 - x
            if target <= 0:
                degrees[x] = 0
                continue
            # Count subsets of [1,100]\{x} that sum to target
            # Use DP to count number of subsets
            other_elements = [i for i in range(1, 101) if i != x]
            dp = [0] * (target + 1)
            dp[0] = 1
            for e in other_elements:
                if e > target:
                    continue
                for s in range(target, e - 1, -1):
                    dp[s] += dp[s - e]
            degrees[x] = dp[target]
        return degrees

    degrees = compute_degrees()

    # Strategy 1: Original descending/ascending
    S1 = try_order(range(100, 0, -1))
    S2 = try_order(range(1, 101))

    # Strategy 2: Sort by degree ascending (lowest degree first - least constrained)
    order_by_degree = sorted(range(1, 101), key=lambda x: degrees[x])
    S3 = try_order(order_by_degree)

    # Strategy 3: Sort by degree descending (highest degree first - most constrained first)
    order_by_degree_desc = sorted(range(1, 101), key=lambda x: -degrees[x])
    S4 = try_order(order_by_degree_desc)

    # Strategy 4: Random orders with degree-weighted priority
    random.seed(42)
    best_rand = set()
    max_iter = 200
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        Sr = try_order(order)
        if len(Sr) > len(best_rand):
            best_rand = Sr

    # Strategy 5: Local improvement - try swapping elements to increase cardinality
    def local_improve(S, max_iters=500):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 5:
                break
            improved = False
            iters += 1
            # Try adding an element not in S
            sums = {0}
            for x in S:
                new_sums = set()
                for s in sums:
                    ns = s + x
                    if ns <= 200:
                        new_sums.add(ns)
                sums.update(new_sums)

            # Find elements that can be added
            addable = []
            for x in range(1, 101):
                if x not in S and (200 - x) not in sums:
                    addable.append(x)

            if not addable:
                break

            # Try removing one element and adding another to increase size
            # Or just add if possible (shouldn't happen since S was built greedily, but check)
            # Actually, try: remove one element from S, see if we can add two new ones
            for remove_x in list(S):
                S_temp = S - {remove_x}
                sums_temp = {0}
                for x in S_temp:
                    new_sums = set()
                    for s in sums_temp:
                        ns = s + x
                        if ns <= 200:
                            new_sums.add(ns)
                    sums_temp.update(new_sums)

                # Count how many new elements can be added
                addable_temp = []
                for x in range(1, 101):
                    if x not in S_temp and (200 - x) not in sums_temp:
                        addable_temp.append(x)

                # If we can add more than 1 new element by removing 1, do it
                if len(addable_temp) > 1:
                    # Add the best new element (lowest degree)
                    addable_temp.sort(key=lambda x: degrees[x])
                    S = S_temp
                    S.add(addable_temp[0])
                    improved = True
                    break

        return S

    candidates = [S1, S2, S3, S4, best_rand]
    best = max(candidates, key=len)
    best = local_improve(best)

    # Try local improvement on all candidates
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        improved_cand = local_improve(cand)
        if len(improved_cand) > len(best):
            best = improved_cand

    return list(best)
