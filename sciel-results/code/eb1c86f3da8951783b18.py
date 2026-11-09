import time
import random

def construct():
    deadline = time.time() + 85.0
    N = 512
    MAXC = 48
    cache = {}

    def build_dp(S):
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def get_dps(S):
        key = tuple(sorted(S))
        r = cache.get(key)
        if r is not None:
            return r
        dp_full = build_dp(S)
        maxfull = sum(S)
        dp_without = {}
        n = len(S)
        for i in range(n):
            sub = S[:i] + S[i + 1:]
            dp_without[S[i]] = (build_dp(sub), maxfull - S[i])
        r = (dp_full, maxfull, dp_without)
        cache[key] = r
        return r

    def test_add(S, x, dp_full, maxfull, dp_without):
        n = len(S)
        # Check 1: x not the mean of any subset of S (size>=2)
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False
        # Check 2: no a in S becomes mean of a subset containing x
        for a in S:
            dpo, maxo = dp_without[a]
            m = n - 1
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                if (dpo[j] >> t) & 1:
                    return False
        return True

    def verify(S):
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    base = list(range(1, N + 1))
    random.seed(20240611)
    orders = [base, base[::-1]]
    for _ in range(12):
        o = base[:]
        random.shuffle(o)
        orders.append(o)

    best = []
    best_sz = 0

    # Phase 1: multi-order greedy
    for order in orders:
        if time.time() > deadline - 8:
            break
        cur = []
        dps = None
        for x in order:
            if len(cur) >= MAXC:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        if verify(cur) and len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
        if best_sz >= MAXC:
            break

    # Phase 2: local search (add + swap) with more iterations
    if best_sz < MAXC and time.time() < deadline - 4:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            dps = get_dps(S)
            # Try adding unused elements
            for u in unused[:]:
                if len(S) >= MAXC:
                    break
                if test_add(S, u, dps[0], dps[1], dps[2]):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    break
            if len(S) >= MAXC:
                break
            # More aggressive swap attempts
            random.shuffle(unused)
            for u in unused[:200]:
                if time.time() > deadline - 2:
                    break
                for i, s in enumerate(S):
                    newS = S[:i] + S[i + 1:]
                    dps2 = get_dps(newS)
                    if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                        S = newS + [u]
                        Sset.remove(s)
                        Sset.add(u)
                        unused.remove(u)
                        unused.append(s)
                        improved = True
                        break
                if improved:
                    break
        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Final safety: trim to a verified valid set (removal preserves validity)
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
