"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 118  # safety margin before 120s evaluator limit
    N = 284
    PRESCRIBED = {1, 2, 5, 11}
    PRESCRIBED_LIST = sorted(PRESCRIBED)

    def compute_sums(S):
        s = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s.add(a + b)
        return s

    def can_add(S, sums, x):
        if x in S:
            return False
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def is_valid_sidon(S):
        """Verify S is a valid Sidon set."""
        if len(S) != len(set(S)):
            return False
        if not PRESCRIBED.issubset(set(S)):
            return False
        if any(x < 1 or x > N for x in S):
            return False
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_with_order(order, S_init):
        """Greedy construction using a specific candidate order."""
        S = list(S_init)
        S.sort()
        sums = compute_sums(S)
        S_set = set(S)
        for x in order:
            if x in S_set:
                continue
            if can_add(S, sums, x):
                S.append(x)
                S_set.add(x)
                for y in S[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        S.sort()
        return S

    def local_search_improved(S, rng, max_rounds=80):
        """Aggressive local search with add, swap, and 2-swap moves."""
        S = list(S)
        S.sort()
        S_set = set(S)
        sums = compute_sums(S)
        best_S = list(S)
        best_size = len(S)
        no_improve_count = 0

        for round_num in range(max_rounds):
            if time.time() - start > DEADLINE:
                break

            improved = False

            # Try to add a new element
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates:
                if time.time() - start > DEADLINE:
                    break
                if can_add(S, sums, x):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    for y in S[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break

            # Try swap: remove one non-prescribed element, add another
            if not improved:
                removable_indices = [i for i, x in enumerate(S) if x not in PRESCRIBED]
                rng.shuffle(removable_indices)
                for i in removable_indices:
                    if time.time() - start > DEADLINE:
                        break
                    removed = S[i]
                    S_without = S[:i] + S[i+1:]
                    S_without.sort()
                    S_without_set = set(S_without)
                    sums_without = compute_sums(S_without)
                    
                    candidates = [x for x in range(1, N + 1) if x not in S_without_set]
                    rng.shuffle(candidates)
                    found = False
                    for x in candidates:
                        if time.time() - start > DEADLINE:
                            break
                        if can_add(S_without, sums_without, x):
                            S = S_without + [x]
                            S.sort()
                            S_set = set(S)
                            sums = compute_sums(S)
                            improved = True
                            found = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = list(S)
                            break
                    if found:
                        break

            # Try 2-swap: remove two non-prescribed elements, add two new ones
            if not improved and len(S) > len(PRESCRIBED):
                removable_indices = [i for i, x in enumerate(S) if x not in PRESCRIBED]
                if len(removable_indices) >= 2:
                    rng.shuffle(removable_indices)
                    # Try a few pairs
                    for _ in range(min(5, len(removable_indices) * (len(removable_indices) - 1) // 2)):
                        if time.time() - start > DEADLINE:
                            break
                        i, j = rng.sample(removable_indices, 2)
                        i, j = sorted([i, j], reverse=True)
                        S_without = S[:i] + S[i+1:j] + S[j+1:]
                        S_without.sort()
                        S_without_set = set(S_without)
                        sums_without = compute_sums(S_without)
                        
                        candidates = [x for x in range(1, N + 1) if x not in S_without_set]
                        rng.shuffle(candidates)
                        added = []
                        found = False
                        for x in candidates:
                            if time.time() - start > DEADLINE:
                                break
                            if can_add(S_without + added, sums_without, x):
                                added.append(x)
                                for y in S_without + added[:-1]:
                                    sums_without.add(x + y)
                                sums_without.add(2 * x)
                                if len(added) == 2:
                                    found = True
                                    break
                        if found:
                            S = S_without + added
                            S.sort()
                            S_set = set(S)
                            sums = compute_sums(S)
                            improved = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = list(S)
                            break

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break

        return best_S

    def simulated_annealing(S, rng, max_iters=3000):
        """Simulated annealing to escape local optima."""
        S = list(S)
        S.sort()
        S_set = set(S)
        sums = compute_sums(S)
        best_S = list(S)
        best_size = len(S)
        temp = 10.0
        cooling = 0.995

        for it in range(max_iters):
            if time.time() - start > DEADLINE:
                break

            # Random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                if can_add(S, sums, x):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    for y in S[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
            elif move_type < 0.7:
                # Try to remove a non-prescribed element
                removable = [x for x in S if x not in PRESCRIBED]
                if not removable:
                    continue
                x = rng.choice(removable)
                S.remove(x)
                S.sort()
                S_set = set(S)
                sums = compute_sums(S)
            else:
                # Swap: remove one, add another
                removable = [x for x in S if x not in PRESCRIBED]
                if not removable:
                    continue
                x = rng.choice(removable)
                S.remove(x)
                S.sort()
                S_set = set(S)
                sums = compute_sums(S)
                
                candidates = [y for y in range(1, N + 1) if y not in S_set]
                if candidates:
                    y = rng.choice(candidates)
                    if can_add(S, sums, y):
                        S.append(y)
                        S.sort()
                        S_set.add(y)
                        for z in S[:-1]:
                            sums.add(y + z)
                        sums.add(2 * y)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = list(S)
                    else:
                        # Revert
                        S.append(x)
                        S.sort()
                        S_set = set(S)
                        sums = compute_sums(S)

            temp *= cooling

        return best_S

    # Strategy: Multiple approaches, keep best
    best_S = list(PRESCRIBED_LIST)
    best_size = len(PRESCRIBED_LIST)

    # 1. Try many greedy constructions with different random orders
    for seed in range(300):
        if time.time() - start > DEADLINE:
            break
        rng = random.Random(seed)
        candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED]
        rng.shuffle(candidates)
        S = greedy_with_order(candidates, PRESCRIBED_LIST)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # 2. Local search on best found
    if time.time() - start < DEADLINE:
        rng = random.Random(12345)
        S = local_search_improved(best_S, rng)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # 3. Try local search on multiple good starting points
    good_starts = []
    for seed in range(80):
        if time.time() - start > DEADLINE:
            break
        rng = random.Random(1000 + seed)
        candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED]
        rng.shuffle(candidates)
        S = greedy_with_order(candidates, PRESCRIBED_LIST)
        if len(S) >= best_size - 2:  # Keep close-to-best starts
            good_starts.append(S)

    for S_start in good_starts[:15]:
        if time.time() - start > DEADLINE:
            break
        rng = random.Random(99999 + len(good_starts))
        S = local_search_improved(S_start, rng)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # 4. Simulated annealing from best
    if time.time() - start < DEADLINE - 10:
        rng = random.Random(77777)
        S = simulated_annealing(best_S, rng, max_iters=4000)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # 5. Final local search polish
    if time.time() - start < DEADLINE - 5:
        rng = random.Random(424242)
        S = local_search_improved(best_S, rng, max_rounds=120)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Verify result
    if not is_valid_sidon(best_S):
        # Fallback to prescribed only if something went wrong
        best_S = list(PRESCRIBED_LIST)

    return best_S
# EVOLVE-BLOCK-END
