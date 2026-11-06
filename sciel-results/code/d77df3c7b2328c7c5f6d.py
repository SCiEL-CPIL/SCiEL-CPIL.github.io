import time
import random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def is_valid(S):
        S_set = set(S)
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                gij = gcd_mat[a][b]
                for k in range(j + 1, L):
                    c = S[k]
                    if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                        return False
        return True

    def can_add(S, x, S_set):
        for i in range(len(S)):
            a = S[i]
            gax = gcd_mat[a][x]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    return False
        return True

    def find_bad_triples(S):
        """Return list of triples (i,j,k) indices into S that form bad triples."""
        L = len(S)
        bads = []
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                gij = gcd_mat[a][b]
                for k in range(j + 1, L):
                    c = S[k]
                    if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                        bads.append((i, j, k))
        return bads

    def simulated_annealing(initial_S, seed_val, max_iters=20000):
        """
        Simulated annealing:
        - State: set S
        - Move: remove one element from a bad triple (if any), or swap an element
        - Accept: always if improves (no bad triples), else with probability exp(-delta/T)
        """
        random.seed(seed_val)
        S = sorted(initial_S[:])
        S_set = set(S)
        remaining = [x for x in UNIVERSE if x not in S_set]

        # Initial temperature
        T = 10.0
        cooling = 0.9995
        best_S = S[:]
        best_len = len(S)
        best_bad_count = len(find_bad_triples(S))

        iters = 0
        while iters < max_iters and time.time() < DEADLINE - 3:
            iters += 1

            bads = find_bad_triples(S)

            if not bads:
                # No bad triples: try to add elements
                if remaining:
                    # Try adding a random element from remaining
                    n_try = min(5, len(remaining))
                    added = False
                    for _ in range(n_try):
                        idx = random.randint(0, len(remaining) - 1)
                        x = remaining[idx]
                        if can_add(S, x, S_set):
                            S.append(x)
                            S.sort()
                            S_set.add(x)
                            remaining.pop(idx)
                            added = True
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                                best_bad_count = 0
                            break
                    if not added:
                        # Try swap
                        if len(S) > 1:
                            i = random.randint(0, len(S) - 1)
                            if remaining:
                                j = random.randint(0, len(remaining) - 1)
                                a = S[i]
                                b = remaining[j]
                                new_S = S[:i] + S[i+1:] + [b]
                                new_S.sort()
                                if is_valid(new_S):
                                    S = new_S
                                    S_set = set(S)
                                    remaining.remove(b)
                                    remaining.append(a)
                                    if len(S) > best_len:
                                        best_S = S[:]
                                        best_len = len(S)
                                        best_bad_count = 0
                # Occasionally cool
                if iters % 100 == 0:
                    T *= cooling
                continue

            # Have bad triples: remove one element from a random bad triple
            bi, bj, bk = random.choice(bads)
            # Choose which of the three to remove (random)
            remove_idx = random.choice([bi, bj, bk])
            removed = S.pop(remove_idx)
            S_set.remove(removed)
            remaining.append(removed)
            remaining.sort()

            # Try to re-add some removed elements
            for _ in range(min(3, len(remaining))):
                idx = random.randint(0, len(remaining) - 1)
                x = remaining[idx]
                if can_add(S, x, S_set):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    remaining.pop(idx)
                    if len(S) > best_len and not find_bad_triples(S):
                        best_S = S[:]
                        best_len = len(S)
                        best_bad_count = 0

            T *= cooling

        # Final: try to clean up
        S_sorted = sorted(S)
        S_set = set(S_sorted)
        remaining = [x for x in UNIVERSE if x not in S_set]
        # Greedy add
        for x in remaining[:]:
            if can_add(S_sorted, x, S_set):
                S_sorted.append(x)
                S_sorted.sort()
                S_set.add(x)
                remaining.remove(x)
        if len(S_sorted) > best_len and is_valid(S_sorted):
            best_S = S_sorted[:]
            best_len = len(S_sorted)

        return best_S

    def greedy_construct(seed_val, order="natural"):
        random.seed(seed_val)
        if order == "natural":
            candidates = list(UNIVERSE)
        elif order == "reverse":
            candidates = list(reversed(UNIVERSE))
        elif order == "random":
            candidates = list(UNIVERSE)
            random.shuffle(candidates)
        elif order == "prime_first":
            def score(x):
                if x == 1:
                    return -1
                spf = x
                d = 2
                while d * d <= x:
                    if x % d == 0:
                        spf = d
                        break
                    d += 1
                return -spf
            candidates = sorted(UNIVERSE, key=score, reverse=True)
        else:
            candidates = list(UNIVERSE)
            random.shuffle(candidates)

        selected = []
        S_set = set()
        for x in candidates:
            if can_add(selected, x, S_set):
                selected.append(x)
                S_set.add(x)
        return selected

    # Multi-start: generate diverse initial solutions via greedy
    initial_solutions = []
    for order in ["natural", "reverse", "prime_first"]:
        initial_solutions.append(greedy_construct(0, order))
    for s in range(1, 20):
        if time.time() >= DEADLINE - 5:
            break
        initial_solutions.append(greedy_construct(s, "random"))

    best_solution = initial_solutions[0]
    best_len = len(best_solution)

    # Run simulated annealing on each initial solution
    for i, init in enumerate(initial_solutions):
        if time.time() >= DEADLINE - 5:
            break
        if len(init) + 5 < best_len:
            continue  # unlikely to beat
        cand = simulated_annealing(init, seed_val=i * 100, max_iters=15000)
        if len(cand) > best_len:
            best_len = len(cand)
            best_solution = cand

    # Final local search / cleanup on best
    if time.time() < DEADLINE - 3:
        final = simulated_annealing(best_solution[:], seed_val=9999, max_iters=20000)
        if len(final) > best_len:
            best_len = len(final)
            best_solution = final

    # Final validity check
    if not is_valid(best_solution):
        # Fallback: simple greedy
        selected = []
        S_set = set()
        for x in UNIVERSE:
            if can_add(selected, x, S_set):
                selected.append(x)
                S_set.add(x)
        if selected:
            best_solution = selected
            best_len = len(selected)
        else:
            best_solution = [1]
            best_len = 1

    return best_solution


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
