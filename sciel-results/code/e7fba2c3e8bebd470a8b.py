"""Optimized: exploit Sidon (B2) structure to maximize |U-U|/|U+U| with small max(U)."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())

def _score_exact(U):
    """Exact score via O(n^2) set computation (fast for n <= ~400)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            sumset.add(ui + U[j])
            diffset.add(ui - U[j])
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom

def _is_sidon(U):
    U = list(U)
    n = len(U)
    seen = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            s = ui + U[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _mian_chowla(n):
    """Greedy Mian-Chowla Sidon sequence (starts 0,1)."""
    if n <= 0:
        return []
    seq = [0, 1] if n >= 2 else [0]
    sums = set()
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            sums.add(seq[i] + seq[j])
    while len(seq) < n:
        cand = seq[-1] + 1
        while True:
            ok = True
            for i in range(len(seq)):
                s = seq[i] + cand
                if s in sums:
                    ok = False
                    break
            if ok:
                seq.append(cand)
                for i in range(len(seq) - 1):
                    sums.add(seq[i] + cand)
                break
            cand += 1
    return seq

def _singer(q):
    """Singer finite-field Sidon set of size q+1 (optimal), q prime power."""
    if q < 2:
        return [0, 1]
    p = q
    e = 1
    while p * p > q:
        p //= 2
        e *= 2
    if p ** e != q:
        return None
    if p == 2:
        def mul(a, b):
            r = 0
            for _ in range(e):
                r ^= a & b
                a = (a >> 1) ^ (a & 1) * (0x12)
                b = (b >> 1)
            return r
    else:
        mod = p ** e
        def mul(a, b):
            r = 0
            for _ in range(e):
                r = (r + a * b) % mod
                a = (a // p) ^ (a % p) * (p - 1)
                b = (b // p)
            return r
    # multiplicative generator
    g = 2
    while True:
        ok = True
        x = 1
        for _ in range(q - 1):
            x = mul(x, g)
            if x == 1:
                ok = False
                break
        if ok:
            break
        g += 1
    powers = []
    x = 1
    for _ in range(q - 1):
        powers.append(x)
        x = mul(x, g)
    S = [0, 1] + [powers[i] - powers[i + 1] for i in range(q - 2)]
    return sorted(set(S))

def _known_sidon(n):
    if n in _MC_CACHE:
        return _MC_CACHE[n]
    if n in _SINGER_CACHE:
        return _SINGER_CACHE[n]
    return None

def _local_search(U, max_iters=20000, deadline=None, allow_break_sidon=True):
    U = sorted(set(U))
    best_score = _score_exact(U)
    best_U = U[:]
    max_u = max(U)
    upper = max_u + 150
    T = 0.02
    min_T = 0.0004
    cooling = 0.9985
    no_improve = 0
    max_no_improve = 25
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if _time_remaining() < 4:
            break
        if no_improve >= max_no_improve and T < min_T:
            break
        improved = False
        idxs = list(range(1, len(U)))
        random.shuffle(idxs)
        for i in idxs:
            old = U[i]
            cands = []
            for off in range(-25, 26):
                nv = old + off
                if nv > 0:
                    cands.append(nv)
            for _ in range(12):
                cands.append(random.randint(1, upper))
            for nv in sorted(cands, key=lambda x: abs(x - old)):
                if nv == old:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                if not allow_break_sidon and not _is_sidon(new_U):
                    continue
                s = _score_exact(new_U)
                d = s - best_score
                if d > 1e-12 or (T > min_T and random.random() < math.exp(d / T)):
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
            if len(U) < 60:
                for _ in range(25):
                    nv = random.randint(1, upper)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    if not allow_break_sidon and not _is_sidon(new_U):
                        continue
                    s = _score_exact(new_U)
                    d = s - best_score
                    if d > 1e-12 or (T > min_T and random.random() < math.exp(d / T)):
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 4:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i + 1:]
                    if not allow_break_sidon and not _is_sidon(new_U):
                        continue
                    s = _score_exact(new_U)
                    d = s - best_score
                    if d > 1e-12 or (T > min_T and random.random() < math.exp(d / T)):
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
        T = max(min_T, T * cooling)
    return sorted(set(best_U)), best_score

_MC_CACHE = {}
_SINGER_CACHE = {}

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 140
    deadline = _DEADLINE
    random.seed(7)

    # Build a pool of candidate Sidon sets (short => small max => best exponent)
    pool = []
    # Mian-Chowla for a range of sizes
    for n in range(3, 60):
        if time.time() > deadline - 100:
            break
        try:
            mc = _mian_chowla(n)
            _MC_CACHE[n] = mc
            pool.append(mc)
        except Exception:
            break
    # Singer sets (optimal short Sidon)
    for q in [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71, 73, 79, 81, 83, 89, 97, 101, 103, 107, 109, 113, 121, 127, 128, 131, 137, 139, 149, 151, 157, 163, 167, 169, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]:
        if time.time() > deadline - 100:
            break
        try:
            s = _singer(q)
            if s is not None and len(s) >= 3:
                _SINGER_CACHE[len(s)] = s
                pool.append(s)
        except Exception:
            continue

    if not pool:
        return [0, 1, 3]

    # Score all candidates
    scored = []
    for U in pool:
        if time.time() > deadline - 80:
            break
        s = _score_exact(U)
        scored.append((s, U))
    scored.sort(key=lambda x: -x[0])
    best_score, best_U = scored[0]

    # Local search on the top candidates (Sidon-preserving, then free)
    for s, U in scored[:25]:
        if time.time() > deadline - 20:
            break
        if len(U) > 400:
            continue
        try:
            u2, s2 = _local_search(U, max_iters=15000, deadline=deadline - 15,
                                    allow_break_sidon=False)
            if s2 > best_score:
                best_score, best_U = s2, u2
        except Exception:
            pass
    # One free local search on the current best (may break Sidon if it helps)
    if time.time() < deadline - 10 and len(best_U) <= 300:
        try:
            u2, s2 = _local_search(best_U, max_iters=20000, deadline=deadline - 6,
                                    allow_break_sidon=True)
            if s2 > best_score:
                best_score, best_U = s2, u2
        except Exception:
            pass
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
