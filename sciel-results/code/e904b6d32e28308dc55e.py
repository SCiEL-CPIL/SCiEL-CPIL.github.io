"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict graph: conflict[a][b] = True if a*b % (a+b) == 0
    # We need a subset S such that for all distinct a,b in S, a*b % (a+b) != 0
    # i.e., (a+b) does NOT divide a*b.

    # Note: a*b % (a+b) == 0 iff (a+b) | a*b
    # Let g = gcd(a,b). Then a = g*a', b = g*b', gcd(a',b')=1
    # a+b = g*(a'+b'), a*b = g^2*a'*b'
    # (a+b) | a*b iff g*(a'+b') | g^2*a'*b' iff (a'+b') | g*a'*b'
    # Since gcd(a'+b', a') = gcd(b', a') = 1 and gcd(a'+b', b') = gcd(a', b') = 1
    # So (a'+b') | g.

    # So conflict exists when (a'+b') divides g, where a = g*a', b = g*b', gcd(a',b')=1.

    # For speed, precompute a conflict bitmask or set.
    # With 256 elements, we can use a 256x256 boolean matrix or bitmasks.

    import math

    # Build conflict set as a list of sets for each number
    conflicts = [set() for _ in range(N + 1)]

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Strategy: Start with a good greedy solution, then improve via local search.
    # The baseline is all odds (128 elements). Let's try to do better.

    # Greedy: sort numbers by number of conflicts (ascending), add if no conflict with current set.
    # Multiple random orderings to find a good initial solution.

    def greedy_build(order):
        selected = set()
        for num in order:
            # Check if num conflicts with any in selected
            if not (conflicts[num] & selected):
                selected.add(num)
        return selected

    def local_search(selected, max_iters=50000):
        """Try to improve by swapping or adding/removing elements."""
        selected = set(selected)
        best_size = len(selected)
        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1

            # Try adding each unselected element
            unselected = [x for x in range(1, N + 1) if x not in selected]
            random.shuffle(unselected)
            for x in unselected:
                if not (conflicts[x] & selected):
                    selected.add(x)
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: remove one selected, add two unselected
            # Or remove one, add one (if it allows adding more later)
            sel_list = list(selected)
            random.shuffle(sel_list)

            for i, rem in enumerate(sel_list[:50]):  # Limit swaps per iteration
                # Try to add unselected elements that conflict with rem
                candidates = [x for x in unselected if x not in selected]
                # Actually, let's try a simpler swap: remove rem, see if we can add more
                selected.remove(rem)
                added_any = False
                for x in unselected:
                    if x not in selected and not (conflicts[x] & selected):
                        selected.add(x)
                        added_any = True
                        break
                if not added_any:
                    # Try adding two
                    added_count = 0
                    for x in unselected:
                        if x not in selected and not (conflicts[x] & selected):
                            selected.add(x)
                            added_count += 1
                            if added_count == 2:
                                break
                    if added_count < 1:
                        selected.add(rem)  # Revert
                    else:
                        improved = True
                        break
                else:
                    improved = True
                    break

            if time.time() > deadline - 10:
                break

        return selected

    # First, try deterministic greedy with different orderings
    best_set = set()

    # Ordering 1: by number of conflicts (ascending)
    conflict_counts = [len(conflicts[i]) for i in range(1, N + 1)]
    order_by_conflicts = sorted(range(1, N + 1), key=lambda x: conflict_counts[x])
    s1 = greedy_build(order_by_conflicts)
    if len(s1) > len(best_set):
        best_set = s1

    # Ordering 2: odd first, then even by conflict count
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    evens_sorted = sorted(evens, key=lambda x: conflict_counts[x])
    order_odds_first = odds + evens_sorted
    s2 = greedy_build(order_odds_first)
    if len(s2) > len(best_set):
        best_set = s2

    # Ordering 3: all odds (baseline)
    s3 = set(odds)
    if len(s3) > len(best_set):
        best_set = s3

    # Try local search from each initial solution
    initial_solutions = [s1, s2, s3]
    for sol in initial_solutions:
        if time.time() > deadline - 15:
            break
        improved_sol = local_search(set(sol), max_iters=10000)
        if len(improved_sol) > len(best_set):
            best_set = improved_sol

    # More random restarts with greedy + local search
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break

        # Random ordering with some structure
        order = list(range(1, N + 1))
        random.shuffle(order)
        sol = greedy_build(order)

        # Quick local search
        improved_sol = local_search(set(sol), max_iters=2000)
        if len(improved_sol) > len(best_set):
            best_set = improved_sol

    # Final: try to add any remaining elements to best_set
    for x in range(1, N + 1):
        if x not in best_set and not (conflicts[x] & best_set):
            best_set.add(x)

    # Verify the solution
    for a in best_set:
        for b in best_set:
            if a != b:
                if (a * b) % (a + b) == 0:
                    # This should not happen; remove b to fix
                    best_set.remove(b)
                    break

    return sorted(best_set)
# EVOLVE-BLOCK-END
