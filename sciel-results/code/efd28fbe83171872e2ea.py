import time

_P = 27720  # lcm(1..12)
_N = 12
_UB = 144   # contract: raw f(12,m) <= 144

def _first_multiple_gt(r, i):
    # smallest multiple of i strictly greater than r
    return (r // i + 1) * i

def _has_matching(r, L):
    # divisors i=1..N; candidates b in (r, r+L] with i|b
    # build for each i the list of candidate b
    # matching: assign distinct b to each i
    # use DFS augmenting (Hopcroft not needed for 12)
    # map each b to a global index
    # We'll build adjacency i -> list of b
    adj = []
    b_to_idx = {}
    idx = 0
    for i in range(1, _N + 1):
        lst = []
        b = _first_multiple_gt(r, i)
        hi = r + L
        while b <= hi:
            if b not in b_to_idx:
                b_to_idx[b] = idx
                idx += 1
            lst.append(b_to_idx[b])
            b += i
        adj.append(lst)
    # match: matchR[bidx] = i (0-based divisor)
    matchR = [-1] * idx
    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if matchR[v] == -1 or dfs(matchR[v], seen):
                matchR[v] = u
                return True
        return False
    cnt = 0
    for u in range(_N):
        seen = [False] * idx
        if dfs(u, seen):
            cnt += 1
    return cnt == _N

def _f_for_residue(r):
    # binary search smallest L in [1, UB] with matching
    lo, hi = 1, _UB
    if not _has_matching(r, _UB):
        # shouldn't happen per contract; return UB
        return _UB
    ans = _UB
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_matching(r, mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    deadline = time.time() + 70.0  # safety margin under 90s
    best_r = 0
    best_f = -1
    # Precompute f for all residues; periodicity means f(12,m) depends only on m mod P.
    # We scan residues; stop early if time is tight (but we keep the best seen).
    for r in range(_P):
        if time.time() > deadline:
            break
        f = _f_for_residue(r)
        if f > best_f:
            best_f = f
            best_r = r
    return int(best_r)
