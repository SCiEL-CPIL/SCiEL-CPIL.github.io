"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    conflict_set = [set() for _ in range(N)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_set[i-1].add(j-1)
                conflict_set[j-1].add(i-1)
    
    def is_independent(s):
        s_set = set(s)
        for v in s:
            if s_set & conflict_set[v]:
                return False
        return True
    
    # Baseline: odds
    baseline = [i for i in range(N) if (i + 1) % 2 == 1]
    assert is_independent(baseline)
    
    best_set = list(baseline)
    best_size = len(baseline)
    
    # Greedy add evens
    current = list(baseline)
    cur_set = set(current)
    for e in range(N):
        if (e + 1) % 2 == 0 and not (conflict_set[e] & cur_set):
            current.append(e)
            cur_set.add(e)
    
    if len(current) > best_size:
        best_set = list(current)
        best_size = len(current)
    
    def local_search(s, tlimit):
        s = list(s)
        s_set = set(s)
        improved = True
        while improved and time.time() < tlimit:
            improved = False
            # Try remove-one-add-two first (increases size)
            for i in range(len(s)):
                if time.time() >= tlimit:
                    return s
                x = s[i]
                wo = s_set - {x}
                cands = [y for y in range(N) if y not in wo and not (conflict_set[y] & wo)]
                for a in range(len(cands)):
                    if time.time() >= tlimit:
                        return s
                    y1 = cands[a]
                    for b in range(a+1, len(cands)):
                        y2 = cands[b]
                        if y2 not in conflict_set[y1]:
                            s = list(wo) + [y1, y2]
                            s_set = set(s)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
            # Try single swaps (maintain size, may enable future improvements)
            if time.time() >= tlimit:
                return s
            for i in range(len(s)):
                if time.time() >= tlimit:
                    return s
                x = s[i]
                wo = s_set - {x}
                for y in range(N):
                    if y in s_set:
                        continue
                    if not (conflict_set[y] & wo):
                        s[i] = y
                        s_set.discard(x)
                        s_set.add(y)
                        improved = True
                        break
                if improved:
                    continue
        return s
    
    if time.time() < deadline - 5.0:
        tl = min(deadline - 5.0, time.time() + 20.0)
        res = local_search(current, tl)
        if len(res) > best_size:
            best_set = list(res)
            best_size = len(res)
    
    # Randomized restarts
    if time.time() < deadline - 8.0:
        random.seed(999)
        for _ in range(200):
            if time.time() >= deadline - 5.0:
                break
            p = list(best_set)
            ps = set(p)
            k = random.randint(1, 5)
            for _ in range(k):
                if not p:
                    break
                idx = random.randint(0, len(p)-1)
                r = p.pop(idx)
                ps.discard(r)
                for _ in range(40):
                    y = random.randint(0, N-1)
                    if y not in ps and not (conflict_set[y] & ps):
                        p.append(y)
                        ps.add(y)
                        break
            tl = min(deadline - 5.0, time.time() + 5.0)
            res = local_search(p, tl)
            if len(res) > best_size:
                best_set = list(res)
                best_size = len(res)
    
    # Validate
    if not is_independent(best_set):
        best_set = baseline
    
    return sorted([i+1 for i in best_set])
# EVOLVE-BLOCK-END
