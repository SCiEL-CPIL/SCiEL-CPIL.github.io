import time
import random

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s kill
    n = 512
    target = 48

    def elapsed_ok():
        return time.time() < deadline

    def _subset_sum_exists(S, k, t):
        m = len(S)
        if k < 0 or k > m or t < 0:
            return False
        dp = [0] * (k + 1)
        dp[0] = 1
        for val in S:
            for c in range(min(k, m), 0, -1):
                dp[c] |= (dp[c - 1] << val)
        return (dp[k] >> t) & 1

    def full_validate(S):
        m = len(S)
        if m == 0:
            return True
        for i, x in enumerate(S):
            others = S[:i] + S[i+1:]
            mo = len(others)
            for k in range(2, mo + 1):
                t = k * x
                if t > mo * n:
                    continue
                if _subset_sum_exists(others, k, t):
                    return False
        return True

    # Cheap violation proxy: count elements that are the average of some PAIR of others.
    # x is a pair-mean iff there exist a,b in S\{x} with a+b == 2x.
    def count_pair_violations(S):
        m = len(S)
        if m < 3:
            return 0
        bad = 0
        for i, x in enumerate(S):
            others = S[:i] + S[i+1:]
            seen = set()
            found = False
            for a in others:
                need = 2 * x - a
                if need in seen and need != a or (need in seen):
                    # need and a are both in others and distinct positions
                    # We need two distinct elements; using a set of values is unsafe if dupes,
                    # but S has distinct elements, so values are unique.
                    if need in seen:
                        found = True
                        break
                seen.add(a)
            if found:
                bad += 1
        return bad

    # Slightly stronger but still cheap: also check triples (size 3) via 2-sum after fixing one.
    # x is a triple-mean iff exist a,b,c in S\{x} with a+b+c==3x.
    # For each a in others, check if 3x-a is a 2-sum of others\{a}. O(m^3) worst, too heavy.
    # Skip triples in inner loop.

    def build_seed(rng):
        # Exponential spread base
        base = []
        val = 2
        while val <= n and len(base) < 18:
            base.append(val)
            val = int(val * 1.7) + rng.randint(1, 4)
        base = sorted(set(base))
        C = list(base)
        candidates = [v for v in range(1, n + 1) if v not in set(base)]
        rng.shuffle(candidates)
        for c in candidates:
            if len(C) >= target:
                break
            C.append(c)
        return C

    def local_search(seed, rng, budget):
        cur = list(seed)
        cur_bad = count_pair_violations(cur)
        best = list(cur)
        best_bad = cur_bad
        T = 1.0
        cooling = 0.9995
        steps = 0
        while elapsed_ok() and steps < budget:
            steps += 1
            m = len(cur)
            r = rng.random()
            if m < target and r < 0.35:
                cand = rng.randint(1, n)
                if cand in cur:
                    continue
                new = cur + [cand]
                new_bad = count_pair_violations(new)
                d = new_bad - cur_bad
                if d <= 0 or rng.random() < 0.1 * T:
                    cur = new
                    cur_bad = new_bad
                    if cur_bad < best_bad:
                        best_bad = cur_bad
                        best = list(cur)
            elif m > 0 and r < 0.7:
                i = rng.randrange(m)
                removed = cur[i]
                cand = rng.randint(1, n)
                if cand == removed or cand in cur:
                    continue
                new = cur[:i] + cur[i+1:] + [cand]
                new_bad = count_pair_violations(new)
                d = new_bad - cur_bad
                if d <= 0 or rng.random() < 0.1 * T:
                    cur = new
                    cur_bad = new_bad
                    if cur_bad < best_bad:
                        best_bad = cur_bad
                        best = list(cur)
            elif m > 12 and r < 0.85:
                i = rng.randrange(m)
                new = cur[:i] + cur[i+1:]
                new_bad = count_pair_violations(new)
                d = new_bad - cur_bad
                if d <= 0 or rng.random() < 0.05 * T:
                    cur = new
                    cur_bad = new_bad
                    if cur_bad < best_bad:
                        best_bad = cur_bad
                        best = list(cur)
            else:
                if m < target:
                    cand = rng.randint(1, n)
                    if cand in cur:
                        continue
                    new = cur + [cand]
                    new_bad = count_pair_violations(new)
                    d = new_bad - cur_bad
                    if d <= 0 or rng.random() < 0.1 * T:
                        cur = new
                        cur_bad = new_bad
                        if cur_bad < best_bad:
                            best_bad = cur_bad
                            best = list(cur)
            T *= cooling
            if T < 0.05:
                T = 1.0
            if best_bad == 0 and len(best) == target:
                return best
        return best

    rng = random.Random(77731)
    best_overall = []
    best_bad_overall = 9999

    # Try multiple seeds
    for trial in range(60):
        if not elapsed_ok():
            break
        seed = build_seed(rng)
        res = local_search(seed, rng, budget=15000)
        if res:
            bb = count_pair_violations(res)
            if bb < best_bad_overall or (bb == best_bad_overall and len(res) > len(best_overall)):
                best_overall = list(res)
                best_bad_overall = bb
        if best_bad_overall == 0 and len(best_overall) == target:
            break

    # Longer polish from best
    if best_overall and best_bad_overall > 0 and elapsed_ok():
        res = local_search(best_overall, rng, budget=60000)
        bb = count_pair_violations(res)
        if bb < best_bad_overall:
            best_overall = list(res)
            best_bad_overall = bb

    # Final full validation; trim if invalid
    if best_overall and not full_validate(best_overall):
        S = list(best_overall)
        changed = True
        while changed and S:
            changed = False
            for i, x in enumerate(S):
                others = S[:i] + S[i+1:]
                mo = len(others)
                is_mean = False
                for k in range(2, mo + 1):
                    t = k * x
                    if t > mo * n:
                        continue
                    if _subset_sum_exists(others, k, t):
                        is_mean = True
                        break
                if is_mean:
                    S.pop(i)
                    changed = True
                    break
        best_overall = S

    if not best_overall:
        fb = [1, 3, 7, 15, 31, 63, 127, 255]
        best_overall = fb if full_validate(fb) else [1]

    return best_overall
