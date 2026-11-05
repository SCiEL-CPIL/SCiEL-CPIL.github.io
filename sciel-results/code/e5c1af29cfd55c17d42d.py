import time
import random

def construct():
    start = time.time()
    DEADLINE = 78  # safety margin before 90s kill

    N = 180

    # Precompute all reciprocal Schur triples (a, b, c) with 1/a = 1/b + 1/c, a,b,c distinct
    # 1/a = 1/b + 1/c => bc = a(b+c) => (b-a)(c-a) = a^2
    # So for each a, factor pairs (d, e) of a^2 with d*e = a^2, d != e
    # then b = a + d, c = a + e, with b,c in [1, N] and all distinct

    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a2 + 1):
            if a2 % d == 0:
                e = a2 // d
                if d == e:
                    continue
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c and b != a and c != a:
                    if a != b and a != c and b != c:
                        triples.append((a, b, c))

    # Deduplicate
    seen = set()
    unique_triples = []
    for t in triples:
        fs = frozenset(t)
        if len(fs) == 3:
            if fs not in seen:
                seen.add(fs)
                unique_triples.append(tuple(sorted(fs)))

    num_triples = len(unique_triples)

    # Build incidence: element -> list of triple indices containing it
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(unique_triples):
        for x in t:
            if x in elem_triples:
                elem_triples[x].append(idx)

    def count_violations(S):
        """For each element in S, count how many triples have all 3 elements in S."""
        in_set = set(S)
        viol = {i: 0 for i in in_set}
        for t in unique_triples:
            if all(x in in_set for x in t):
                for x in t:
                    viol[x] += 1
        return viol

    def is_valid(S):
        in_set = set(S)
        for t in unique_triples:
            if all(x in in_set for x in t):
                return False
        return True

    def fix_by_removal(S):
        """Remove minimal elements to make S valid. Greedy: remove element in most violated triples."""
        changed = True
        while changed:
            changed = False
            in_set = set(S)
            viol = {i: 0 for i in in_set}
            for t in unique_triples:
                if all(x in in_set for x in t):
                    for x in t:
                        viol[x] += 1
            max_v = 0
            worst = None
            for x in in_set:
                v = viol.get(x, 0)
                if v > max_v:
                    max_v = v
                    worst = x
            if worst is not None and max_v > 0:
                S.discard(worst)
                changed = True
        return S

    # Strategy: Start from the known-good set {73..179} (size 107) and try to grow it
    # by adding elements from {1..72} and {180} (if not already included)
    
    # Actually, let's start with the full set [1..180] and use a smarter approach:
    # 1. Start with the known good interval [73..179]
    # 2. Try to add elements from the lower range [1..72] one by one
    # 3. For each addition, fix violations by removing elements
    # 4. Use multiple passes and random restarts

    # Start with the known good set
    base_S = set(range(73, 180))  # {73, ..., 179}, size 107
    best_S = set(base_S)
    best_size = len(best_S)

    # Elements not in base_S
    excluded = [i for i in range(1, N + 1) if i not in base_S]

    # Pass 1: Try adding each excluded element
    for x in excluded:
        if time.time() > DEADLINE:
            break
        trial = set(best_S)
        trial.add(x)
        fixed = fix_by_removal(trial)
        if len(fixed) > best_size:
            best_S = fixed
            best_size = len(fixed)

    # Pass 2: Multiple rounds of hill climbing
    for _ in range(30):
        if time.time() > DEADLINE:
            break
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        improved = False
        for x in excluded:
            if time.time() > DEADLINE:
                break
            trial = set(best_S)
            trial.add(x)
            fixed = fix_by_removal(trial)
            if len(fixed) > best_size:
                best_S = fixed
                best_size = len(fixed)
                improved = True
        if not improved:
            break

    # Simulated annealing / random search if time permits
    rng = random.Random(42)
    for iteration in range(500):
        if time.time() > DEADLINE:
            break
        if best_size >= N:
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

    # Also try a different starting point: greedy from full set
    if time.time() < DEADLINE - 10:
        S_full = set(range(1, N + 1))
        S_greedy = fix_by_removal(S_full)
        if len(S_greedy) > best_size:
            best_S = S_greedy
            best_size = len(S_greedy)
            # Continue hill climbing from this better starting point
            for _ in range(20):
                if time.time() > DEADLINE:
                    break
                excluded = [i for i in range(1, N + 1) if i not in best_S]
                improved = False
                for x in excluded:
                    if time.time() > DEADLINE:
                        break
                    trial = set(best_S)
                    trial.add(x)
                    fixed = fix_by_removal(trial)
                    if len(fixed) > best_size:
                        best_S = fixed
                        best_size = len(fixed)
                        improved = True
                if not improved:
                    break

    # Final verification and cleanup
    final = sorted(best_S)
    if not final:
        final = list(range(73, 180))

    # Double-check validity
    in_final = set(final)
    for t in unique_triples:
        if all(x in in_final for x in t):
            final.remove(t[0])
            in_final.discard(t[0])

    if not final:
        final = [1]

    return final
