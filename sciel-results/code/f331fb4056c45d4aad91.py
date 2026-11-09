"""Constructive genetic search over parametric families of finite sets."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _set_sum_diff(U):
    """Return (|U+U|, |U-U|) for a finite set U of ints."""
    Us = sorted(U)
    n = len(Us)
    if n == 0:
        return 0, 0
    plus = set()
    minus = set()
    for i in range(n):
        a = Us[i]
        for j in range(i, n):
            plus.add(a + Us[j])
        for j in range(n):
            minus.add(a - Us[j])
    return len(plus), len(minus)

def _score(U):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
    nplus, nminus = _set_sum_diff(U)
    if nplus == 0:
        return 0.0, sorted(U)
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0, sorted(U)
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin for evaluator
    rng = random.Random(12345)

    best_val = -1e9
    best_set = [0, 1, 3]

    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _score(U)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_set = s

    # 1. Baseline exhaustive search for small sets (quick win)
    for n in range(2, 7):
        for combo in combinations(range(1, 15), n - 1):
            U = [0] + list(combo)
            consider(U)

    # 2. Parametric Family Generator
    # Represent a family by (type, params, n)
    # type: 'poly', 'exp', 'binom', 'fib', 'tri', 'quad'
    def gen_family(fam):
        ftype = fam[0]
        params = fam[1]
        n = fam[2]
        U = [0]
        try:
            if ftype == 'poly':
                # U_k = a*k^d + b*k + c
                d = params[0]
                a = params[1]
                b = params[2]
                c = params[3]
                for k in range(1, n):
                    val = a * (k ** d) + b * k + c
                    if val < 0:
                        val = -val
                    U.append(val)
            elif ftype == 'exp':
                # U_k = a * base^k + b
                base = params[0]
                a = params[1]
                b = params[2]
                for k in range(1, n):
                    val = a * (base ** k) + b
                    if val < 0:
                        val = -val
                    U.append(val)
            elif ftype == 'binom':
                # U_k = C(k + m, m) * a + b
                m = params[0]
                a = params[1]
                b = params[2]
                for k in range(1, n):
                    val = a * math.comb(k + m, m) + b
                    if val < 0:
                        val = -val
                    U.append(val)
            elif ftype == 'fib':
                # U_k = a * F_{k+b} + c
                a = params[0]
                b = params[1]
                c = params[2]
                # Precompute fibs up to n+b
                fibs = [0, 1]
                for i in range(2, n + b + 2):
                    fibs.append(fibs[-1] + fibs[-2])
                for k in range(1, n):
                    val = a * fibs[k + b] + c
                    if val < 0:
                        val = -val
                    U.append(val)
            elif ftype == 'tri':
                # U_k = a * T_k + b
                a = params[0]
                b = params[1]
                for k in range(1, n):
                    val = a * (k * (k + 1) // 2) + b
                    if val < 0:
                        val = -val
                    U.append(val)
            elif ftype == 'quad':
                # U_k = a*k^2 + b*k + c
                a = params[0]
                b = params[1]
                c = params[2]
                for k in range(1, n):
                    val = a * k * k + b * k + c
                    if val < 0:
                        val = -val
                    U.append(val)
            else:
                return None
        except Exception:
            return None

        # Ensure uniqueness and non-negative
        U = list(set(U))
        U = [x for x in U if x >= 0]
        if 0 not in U:
            U.append(0)
        if len(U) < 2:
            return None
        return sorted(U)

    # 3. Initial Population of Families
    pop = []
    # Hand-crafted good seeds
    seeds = [
        ('exp', [2, 1, 0], 10),  # Powers of 2
        ('exp', [3, 1, 0], 8),   # Powers of 3
        ('exp', [4, 1, 0], 7),   # Powers of 4
        ('poly', [2, 1, 0, 0], 10), # Squares
        ('poly', [3, 1, 0, 0], 8),  # Cubes
        ('tri', [1, 0], 10),    # Triangular
        ('quad', [1, 0, 0], 10), # Squares (redundant with poly but different params)
        ('binom', [1, 1, 0], 10), # k+1
        ('binom', [2, 1, 0], 8),  # C(k+2, 2)
        ('fib', [1, 0, 0], 12),   # Fibonacci
    ]
    for s in seeds:
        pop.append(list(s))

    # Random initial families
    for _ in range(20):
        ftype = rng.choice(['poly', 'exp', 'binom', 'fib', 'tri', 'quad'])
        if ftype == 'poly':
            d = rng.randint(1, 4)
            a = rng.randint(1, 5)
            b = rng.randint(-2, 2)
            c = rng.randint(-2, 2)
            n = rng.randint(5, 15)
            pop.append([ftype, [d, a, b, c], n])
        elif ftype == 'exp':
            base = rng.randint(2, 6)
            a = rng.randint(1, 3)
            b = rng.randint(-1, 1)
            n = rng.randint(5, 12)
            pop.append([ftype, [base, a, b], n])
        elif ftype == 'binom':
            m = rng.randint(1, 3)
            a = rng.randint(1, 3)
            b = rng.randint(-1, 1)
            n = rng.randint(5, 12)
            pop.append([ftype, [m, a, b], n])
        elif ftype == 'fib':
            a = rng.randint(1, 3)
            b = rng.randint(0, 3)
            c = rng.randint(-1, 1)
            n = rng.randint(5, 15)
            pop.append([ftype, [a, b, c], n])
        elif ftype == 'tri':
            a = rng.randint(1, 5)
            b = rng.randint(-2, 2)
            n = rng.randint(5, 15)
            pop.append([ftype, [a, b], n])
        elif ftype == 'quad':
            a = rng.randint(1, 5)
            b = rng.randint(-2, 2)
            c = rng.randint(-2, 2)
            n = rng.randint(5, 15)
            pop.append([ftype, [a, b, c], n])

    # Evaluate initial population
    fam_scores = []
    for fam in pop:
        U = gen_family(fam)
        if U:
            v, s = _score(U)
            fam_scores.append((v, fam, s))
            consider(U)

    fam_scores.sort(key=lambda x: -x[0])
    pop = [f[1] for f in fam_scores[:20]]  # Keep top 20

    # 4. Genetic Evolution Loop
    while time.time() < deadline - 2.0:
        new_pop = []
        # Elitism: keep best
        new_pop.append(pop[0])
        if len(pop) > 1:
            new_pop.append(pop[1])

        # Fill rest with offspring
        while len(new_pop) < 20:
            # Crossover/Mutation
            parent1 = pop[rng.randint(0, len(pop) - 1)]
            parent2 = pop[rng.randint(0, len(pop) - 1)]

            # Simple mutation: pick a random parameter and tweak it
            child = [parent1[0], list(parent1[1]), parent1[2]]
            # Mutate type with low probability
            if rng.random() < 0.1:
                child[0] = rng.choice(['poly', 'exp', 'binom', 'fib', 'tri', 'quad'])
                # Reset params to reasonable defaults for new type
                if child[0] == 'poly':
                    child[1] = [rng.randint(1, 4), rng.randint(1, 5), rng.randint(-2, 2), rng.randint(-2, 2)]
                elif child[0] == 'exp':
                    child[1] = [rng.randint(2, 6), rng.randint(1, 3), rng.randint(-1, 1)]
                elif child[0] == 'binom':
                    child[1] = [rng.randint(1, 3), rng.randint(1, 3), rng.randint(-1, 1)]
                elif child[0] == 'fib':
                    child[1] = [rng.randint(1, 3), rng.randint(0, 3), rng.randint(-1, 1)]
                elif child[0] == 'tri':
                    child[1] = [rng.randint(1, 5), rng.randint(-2, 2)]
                elif child[0] == 'quad':
                    child[1] = [rng.randint(1, 5), rng.randint(-2, 2), rng.randint(-2, 2)]
            else:
                # Mutate params
                if child[1]:
                    idx = rng.randint(0, len(child[1]) - 1)
                    # Adjust range based on type
                    if child[0] == 'exp' and idx == 0:
                        child[1][idx] = max(2, min(10, child[1][idx] + rng.randint(-2, 2)))
                    elif child[0] == 'poly' and idx == 0:
                        child[1][idx] = max(1, min(5, child[1][idx] + rng.randint(-1, 1)))
                    else:
                        child[1][idx] += rng.randint(-2, 2)
                        if child[0] == 'exp' and idx == 0:
                            child[1][idx] = max(2, child[1][idx])
                        if child[0] == 'poly' and idx == 0:
                            child[1][idx] = max(1, child[1][idx])

            # Mutate n
            if rng.random() < 0.3:
                child[2] = max(4, min(20, child[2] + rng.randint(-2, 2)))

            # Evaluate
            U = gen_family(child)
            if U:
                v, s = _score(U)
                consider(U)
                new_pop.append(child)
            else:
                # If invalid, try parent again or skip
                new_pop.append(parent1)

        # Sort and select
        scored_new = []
        for fam in new_pop:
            U = gen_family(fam)
            if U:
                v, s = _score(U)
                scored_new.append((v, fam))
        scored_new.sort(key=lambda x: -x[0])
        pop = [f[1] for f in scored_new[:20]]

    return best_set
# EVOLVE-BLOCK-END
