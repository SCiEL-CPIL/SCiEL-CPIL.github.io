import time
import random

def construct():
    start = time.time()
    DEADLINE = 82  # safety margin before 90s kill
    N = 180

    # Precompute all reciprocal Schur triples (a, b, c) with 1/a = 1/b + 1/c,
    # a,b,c distinct.  1/a = 1/b + 1/c  =>  (b-a)(c-a) = a^2.
    # For each a, factor pairs (d, e) of a^2 with d*e = a^2, d != e,
    # then b = a + d, c = a + e (and the swap), with b,c in [1, N], all distinct.
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(a2 ** 0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                if d == e:
                    continue
                b1 = a + d
                c1 = a + e
                if b1 <= N and c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    triples.append((a, b1, c1))
                b2 = a + e
                c2 = a + d
                if b2 <= N and c2 <= N and b2 != c2 and b2 != a and c2 != a:
                    if (a, b2, c2) != (a, b1, c1):
                        triples.append((a, b2, c2))

    # Deduplicate (keep a as the smallest element)
    seen = set()
    unique_triples = []
    for t in triples:
        if t[0] < t[1] and t[0] < t[2]:
            fs = frozenset(t)
            if len(fs) == 3 and fs not in seen:
                seen.add(fs)
                unique_triples.append(tuple(sorted(fs)))

    # ---- Global fixer: remove the element in the most violated triples ----
    def fix_by_removal(S):
        changed = True
        while changed:
            changed = False
            in_set = set(S)
            viol = {}
            for t in unique_triples:
                if t[0] in in_set and t[1] in in_set and t[2] in in_set:
                    for x in t:
                        viol[x] = viol.get(x, 0) + 1
            if not viol:
                break
            worst = None
            max_v = 0
            for x, v in viol.items():
                if v > max_v:
                    max_v = v
                    worst = x
            if worst is not None:
                S.discard(worst)
                changed = True
        return S

    def is_valid(S):
        in_set = set(S)
        for t in unique_triples:
            if t[0] in in_set and t[1] in in_set and t[2] in in_set:
                return False
        return True

    best_S = set()
    best_size = 0

    def consider(S):
        nonlocal best_S, best_size
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)

    # ---- Strategy A: greedy removal from the full set ----
    S_full = set(range(1, N + 1))
    S_greedy = fix_by_removal(S_full)
    consider(S_greedy)

    # ---- Strategy B: interval base + hill climb adding small numbers ----
    # The structural property: only the minimum element of a triple can be 'a'.
    # A large base interval [91,180] is conflict-free internally (a would need
    # to be < 91), so we only lose small numbers that conflict with the base.
    for base_start in (90, 85, 80, 75, 70):
        if time.time() > DEADLINE:
            break
        base = set(range(base_start + 1, N + 1))
        S = set(base)
        # Greedy-add small numbers that don't create a violation
        for a in range(1, base_start + 1):
            if time.time() > DEADLINE:
                break
            trial = set(S)
            trial.add(a)
            if is_valid(trial):
                S = trial
        # Hill climb: try adding each excluded element, fixing if needed
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            excluded = [i for i in range(1, N + 1) if i not in S]
            if not excluded:
                break
            improved = False
            for x in excluded:
                if time.time() > DEADLINE:
                    break
                trial = set(S)
                trial.add(x)
                fixed = fix_by_removal(trial)
                if len(fixed) > len(S):
                    S = fixed
                    improved = True
            if not improved:
                break
        consider(S)

    # ---- Strategy C: multi-restart random init + hill climb ----
    rng = random.Random(42)
    for restart in range(8):
        if time.time() > DEADLINE:
            break
        random_set = set(rng.sample(range(1, N + 1), int(N * 0.7)))
        fixed = fix_by_removal(random_set)
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            excluded = [i for i in range(1, N + 1) if i not in fixed]
            if not excluded:
                break
            improved = False
            for x in excluded:
                if time.time() > DEADLINE:
                    break
                trial = set(fixed)
                trial.add(x)
                fixed_trial = fix_by_removal(trial)
                if len(fixed_trial) > len(fixed):
                    fixed = fixed_trial
                    improved = True
            if not improved:
                break
        consider(fixed)

    # ---- Strategy D: random hill climb on the current best ----
    for iteration in range(3000):
        if time.time() > DEADLINE:
            break
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        if not excluded:
            break
        x = rng.choice(excluded)
        trial = set(best_S)
        trial.add(x)
        fixed = fix_by_removal(trial)
        if len(fixed) > best_size:
            best_S = fixed
            best_size = len(fixed)

    # ---- Final verification and cleanup ----
    final = sorted(best_S)
    if not final:
        final = list(range(91, N + 1))

    in_final = set(final)
    for t in unique_triples:
        if t[0] in in_final and t[1] in in_final and t[2] in in_final:
            final.remove(t[0])
            in_final.discard(t[0])

    if not final:
        final = [1]

    return final
