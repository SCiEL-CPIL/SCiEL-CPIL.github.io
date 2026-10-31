import time
import math
import random
import itertools

def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = set(U)
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_plus = {a + b for a in U for b in U}
    U_minus = {a - b for a in U for b in U}
    val = len(U_minus) / len(U_plus)
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def _sumset(U):
    return {a + b for a in U for b in U}

def _diffset(U):
    return {a - b for a in U for b in U}

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    best_set = [0, 1, 3]
    best_score = _score(best_set)

    def try_set(U):
        nonlocal best_set, best_score
        U = sorted(set(U))
        if 0 not in U or len(U) < 2 or max(U) == 0:
            return
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]

    # ---- Phase 1: Greedy ratio-based construction ----
    # Build sets by greedily adding the element that most improves the score.
    # This directly targets the objective: minimize |U+U| growth while maximizing |U-U|.
    def greedy_construct(max_val, start_set=None, max_size=15):
        if start_set is None:
            U = [0, 1]
        else:
            U = sorted(set(start_set))
        if 0 not in U:
            U = [0] + U
        U = sorted(U)
        current_score = _score(U)
        for _ in range(max_size - len(U)):
            if time_left() < 2:
                break
            best_add = None
            best_add_score = current_score
            # Candidate pool: existing range plus a bit beyond
            upper = max(U) + max(5, max_val // max(1, len(U)))
            candidates = [x for x in range(1, min(upper, max_val) + 1) if x not in U]
            for c in candidates:
                if time_left() < 1:
                    break
                U_new = U + [c]
                s = _score(U_new)
                if s > best_add_score:
                    best_add_score = s
                    best_add = c
            if best_add is None:
                break
            U = U + [best_add]
            U = sorted(U)
            current_score = best_add_score
        return U, current_score

    # Run greedy from several seeds
    random.seed(42)
    greedy_seeds = [
        [0, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 1, 4],
        [0, 3, 7],
        [0, 1, 5],
    ]
    for seed_set in greedy_seeds:
        if time_left() < 5:
            break
        for max_val in [20, 40, 80, 150, 300]:
            if time_left() < 3:
                break
            U_g, s_g = greedy_construct(max_val, seed_set, max_size=12)
            try_set(U_g)

    # ---- Phase 2: Structured families ----
    # Powers
    for base in range(2, 12):
        for k in range(2, 10):
            if time_left() < 2:
                break
            U = [base ** i for i in range(k)]
            if max(U) <= 500:
                try_set(U)

    # Fibonacci
    fib = [0, 1]
    while fib[-1] <= 500:
        fib.append(fib[-1] + fib[-2])
    for n in range(3, len(fib)):
        try_set(fib[:n])

    # Small exhaustive search
    for n in range(2, 7):
        if time_left() < 10:
            break
        for combo in itertools.combinations(range(1, 25), n - 1):
            if time_left() < 5:
                break
            U = [0] + list(combo)
            try_set(U)

    # Arithmetic progressions with perturbations
    for max_u in range(5, 100, 5):
        if time_left() < 5:
            break
        for n in range(3, 10):
            if (max_u) % (n - 1) == 0:
                step = max_u // (n - 1)
                if step >= 1:
                    ap = [i * step for i in range(n)]
                    try_set(ap)

    # ---- Phase 3: Local search with wider moves (add, remove, replace, 2-swap) ----
    def local_search(U, max_iters=3000):
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        current_score = _score(U)
        no_improve = 0
        max_no_improve = 150

        for _ in range(max_iters):
            if time_left() < 5:
                break
            if no_improve >= max_no_improve:
                break
            improved = False
            max_u = max(U)
            search_range = max_u + 20  # wider range

            # Move 1: Add an element
            if len(U) < 18:
                for new_el in range(1, min(search_range, 400) + 1):
                    if new_el not in U:
                        U_new = U + [new_el]
                        s = _score(U_new)
                        if s > current_score:
                            U = U_new
                            current_score = s
                            improved = True
                            break

            # Move 2: Remove an element
            if not improved and len(U) > 3:
                for i, el in enumerate(U):
                    if el == 0:
                        continue
                    U_new = U[:i] + U[i+1:]
                    if len(U_new) >= 2:
                        s = _score(U_new)
                        if s > current_score:
                            U = U_new
                            current_score = s
                            improved = True
                            break

            # Move 3: Replace an element
            if not improved:
                for i, el in enumerate(U):
                    if el == 0:
                        continue
                    for new_el in range(1, min(search_range, 400) + 1):
                        if new_el not in U:
                            U_new = U[:]
                            U_new[i] = new_el
                            U_new = sorted(set(U_new))
                            if 0 not in U_new or len(U_new) < 2:
                                continue
                            s = _score(U_new)
                            if s > current_score:
                                U = U_new
                                current_score = s
                                improved = True
                                break
                    if improved:
                        break

            # Move 4: 2-swap (remove one, add one) - wider neighborhood
            if not improved and len(U) > 3:
                for i, el in enumerate(U):
                    if el == 0:
                        continue
                    U_without = U[:i] + U[i+1:]
                    for new_el in range(1, min(search_range, 400) + 1):
                        if new_el not in U_without:
                            U_new = U_without + [new_el]
                            U_new = sorted(set(U_new))
                            if 0 not in U_new or len(U_new) < 2:
                                continue
                            s = _score(U_new)
                            if s > current_score:
                                U = U_new
                                current_score = s
                                improved = True
                                break
                    if improved:
                        break

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return U, current_score

    # Local search from best set
    U_ls, s_ls = local_search(best_set, max_iters=4000)
    if s_ls > best_score:
        best_set = U_ls
        best_score = s_ls

    # ---- Phase 4: Multiple random restarts with local search ----
    random.seed(123)
    for trial in range(30):
        if time_left() < 15:
            break
        # Diverse random starts
        size = random.randint(3, 14)
        max_el = random.randint(10, 200)
        U_rand = [0] + random.sample(range(1, max_el + 1), min(size - 1, max_el))
        U_rand = sorted(U_rand)
        U_ls, s_ls = local_search(U_rand, max_iters=1500)
        if s_ls > best_score:
            best_set = U_ls
            best_score = s_ls

    # ---- Phase 5: Greedy refinement from best set ----
    if time_left() > 20:
        U_g, s_g = greedy_construct(500, best_set, max_size=len(best_set) + 5)
        if s_g > best_score:
            best_set = U_g
            best_score = s_g
        # Local search after greedy
        U_ls, s_ls = local_search(best_set, max_iters=2000)
        if s_ls > best_score:
            best_set = U_ls
            best_score = s_ls

    # ---- Phase 6: Final polish with tighter moves ----
    if time_left() > 10:
        U_final, s_final = local_search(best_set, max_iters=1000)
        if s_final > best_score:
            best_set = U_final
            best_score = s_final

    return sorted(best_set)
