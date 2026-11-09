import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))
    odds = [x for x in all_nums if x % 2 == 1]
    evens = [x for x in all_nums if x % 2 == 0]

    # Precompute conflict: conflict[e][o] = True if (e*o) % (e+o) == 0
    # Only need even-odd conflicts (odd-odd always compatible, even-even need checking too)
    # Actually we need all pairs. Let's precompute a full conflict matrix for speed.
    # conflict[i][j] = True if i and j are INCOMPATIBLE (i.e., (i*j)%(i+j)==0)
    
    # For memory efficiency with 256x256, use a list of sets
    # conflict_sets[i] = set of j != i such that (i*j)%(i+j) == 0
    conflict_sets = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)

    def is_valid(S):
        """Check if set S is valid (all distinct pairs compatible)."""
        S_list = list(S)
        for idx, a in enumerate(S_list):
            for b in S_list[idx+1:]:
                if b in conflict_sets[a]:
                    return False
        return True

    def greedy_add(S, candidates, rng):
        """Greedily add candidates to S in random order if compatible."""
        S = set(S)
        order = candidates[:]
        rng.shuffle(order)
        for e in order:
            if e in S:
                continue
            # Check if e is compatible with all in S
            compatible = True
            for s in S:
                if e in conflict_sets[s]:
                    compatible = False
                    break
            if compatible:
                S.add(e)
        return S

    def local_search(S, rng, max_iters=200):
        """Try swap moves: remove one element, add another if it improves or maintains size."""
        S = set(S)
        current_size = len(S)
        best_S = set(S)
        best_size = current_size

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Try swap: pick a random element from S to remove, try to add a random element not in S
            S_list = list(S)
            if not S_list:
                break
            remove_elem = rng.choice(S_list)
            S.remove(remove_elem)

            # Try to add a random element from [1,256] not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                S.add(remove_elem)
                continue

            added = False
            # Try a few random candidates
            rng.shuffle(candidates)
            for add_elem in candidates[:20]:  # Try up to 20 random candidates
                compatible = True
                for s in S:
                    if add_elem in conflict_sets[s]:
                        compatible = False
                        break
                if compatible:
                    S.add(add_elem)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break

            if not added:
                S.add(remove_elem)

        return best_S

    # Multi-random greedy approach
    best_solution = set()
    best_size = 0

    num_trials = 500
    for trial in range(num_trials):
        if time.time() > deadline - 5:  # Leave time for local search
            break

        rng = random.Random(trial * 7919 + 13)

        # Start with all odds
        S = set(odds)

        # Add compatible evens via greedy
        S = greedy_add(S, evens, rng)

        size = len(S)
        if size > best_size:
            best_size = size
            best_solution = set(S)

        # Local search to improve
        S_ls = local_search(S, rng, max_iters=50)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_solution = set(S_ls)

    # Final verification
    if not is_valid(best_solution):
        # Fallback: just return all odds
        best_solution = set(odds)

    return list(best_solution)
