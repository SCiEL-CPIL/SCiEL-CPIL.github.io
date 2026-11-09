import time
import random

def construct():
    deadline = time.time() + 87.0
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
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False
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

    # Phase 1: Multi-order greedy to get a good starting set
    base = list(range(1, N + 1))
    random.seed(20240611)
    orders = [base, base[::-1]]
    for _ in range(8):
        o = base[:]
        random.shuffle(o)
        orders.append(o)

    best = []
    best_sz = 0

    for order in orders:
        if time.time() > deadline - 20:
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

    if best_sz >= MAXC:
        return best

    # Phase 2: Simulated Annealing local search
    if time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        # SA parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        sa_start_time = time.time()
        sa_budget = deadline - 5 - sa_start_time
        
        best_sa = S[:]
        best_sa_sz = len(S)
        
        while T > T_min and time.time() < deadline - 3:
            # Randomly pick a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and len(S) < MAXC and unused:
                # Try to add a random unused element
                u = random.choice(unused)
                dps = get_dps(S)
                if test_add(S, u, dps[0], dps[1], dps[2]):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    if len(S) > best_sa_sz:
                        best_sa_sz = len(S)
                        best_sa = S[:]
                    T *= cooling
                else:
                    # Reject with probability based on T
                    if random.random() < 0.1:
                        pass  # Just don't add
                    T *= cooling
                    
            elif move_type < 0.5 and len(S) > 10:
                # Try to remove a random element (to explore smaller sets)
                i = random.randint(0, len(S) - 1)
                removed = S.pop(i)
                Sset.remove(removed)
                unused.append(removed)
                # Accept removal if it helps (or with some probability)
                if random.random() < 0.3 or len(S) < best_sa_sz - 2:
                    T *= cooling
                else:
                    # Undo removal
                    S.append(removed)
                    Sset.add(removed)
                    unused.remove(removed)
                    T *= cooling
                    
            else:
                # Swap: remove one from S, add one from unused
                if unused and len(S) > 0:
                    i = random.randint(0, len(S) - 1)
                    s_old = S[i]
                    u = random.choice(unused)
                    
                    # Try the swap
                    newS = S[:i] + S[i + 1:]
                    dps2 = get_dps(newS)
                    if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
                        S = newS + [u]
                        Sset.remove(s_old)
                        Sset.add(u)
                        unused.remove(u)
                        unused.append(s_old)
                        if len(S) > best_sa_sz:
                            best_sa_sz = len(S)
                            best_sa = S[:]
                        T *= cooling
                    else:
                        # With small probability, accept a "bad" swap to escape local optima
                        if random.random() < T * 0.05:
                            S = newS + [u]
                            Sset.remove(s_old)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s_old)
                        T *= cooling
        
        # After SA, try to fill up with greedy additions
        if best_sa_sz < MAXC and time.time() < deadline - 2:
            S = best_sa[:]
            Sset = set(S)
            unused = [x for x in range(1, N + 1) if x not in Sset]
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                dps = get_dps(S)
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
                # Swap attempts
                random.shuffle(unused)
                for u in unused[:80]:
                    if time.time() > deadline - 1:
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
            if verify(S) and len(S) > best_sa_sz:
                best_sa_sz = len(S)
                best_sa = S[:]
        
        if verify(best_sa) and len(best_sa) > best_sz:
            best_sz = len(best_sa)
            best = best_sa[:]

    # Final safety: trim to a verified valid set
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
