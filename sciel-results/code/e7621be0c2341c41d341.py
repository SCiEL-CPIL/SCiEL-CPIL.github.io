"""Sums and differences of finite sets: maximize 1 + log(|U-U|/|U+U|)/log(2*maxU+1)."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

_START = time.time()

def _time_left(deadline):
    return deadline - time.time()

def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def _sumset_size_fft(U):
    max_u = max(U)
    sum_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * sum_len)
    ind = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        ind[u] = 1.0
    f = fft(ind)
    conv = ifft(f * f).real
    return int(np.sum(conv[:sum_len] > 1e-6)), max_u

def _diffset_size_fft(U):
    max_u = max(U)
    diff_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * diff_len)
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u + max_u] = 1.0
    f = fft(vec)
    res = ifft(f * np.conj(f)).real
    return int(np.sum(np.abs(res[:diff_len]) > 1e-6)), max_u

def _score_set_fft(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    ss, _ = _sumset_size_fft(U)
    ds, _ = _diffset_size_fft(U)
    if ss == 0:
        return 0.0
    ratio = ds / ss
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _score_set_exact(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    ss = set()
    ds = set()
    for i in range(n):
        for j in range(n):
            ss.add(U[i] + U[j])
            ds.add(U[i] - U[j])
    ratio = len(ds) / len(ss)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _score_set(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    if n <= 28:
        return _score_set_exact(U)
    return _score_set_fft(U)

def _bose_sidon(q):
    """Bose Sidon set from finite field GF(q^2) with q odd prime power."""
    if q < 3:
        return [0]
    def is_square(a, p):
        if a == 0:
            return True
        return pow(a, (p - 1) // 2, p) == 1
    c = 2
    while is_square(c % q, q):
        c += 1
    c = c % q
    def fm(a, b):
        a0, a1 = a
        b0, b1 = b
        r0 = (a0 * b0 + c * a1 * b1) % q
        r1 = (a0 * b1 + a1 * b0) % q
        return (r0, r1)
    order = q * q - 1
    factors = []
    d = order
    p = 2
    while p * p <= d:
        if d % p == 0:
            factors.append(p)
            while d % p == 0:
                d //= p
        p += 1
    if d > 1:
        factors.append(d)
    def fpow(base, e):
        res = (1, 0)
        b = base
        while e > 0:
            if e & 1:
                res = fm(res, b)
            b = fm(b, b)
            e >>= 1
        return res
    def feq(x, y):
        return x[0] == y[0] and x[1] == y[1]
    prim = None
    for a0 in range(q):
        for a1 in range(q):
            if a0 == 0 and a1 == 0:
                continue
            e = (a0, a1)
            is_prim = True
            for f in factors:
                if feq(fpow(e, order // f), (1, 0)):
                    is_prim = False
                    break
            if is_prim:
                prim = e
                break
        if prim is not None:
            break
    if prim is None:
        return [0, 1, 3]
    elems = []
    cur = (1, 0)
    for i in range(q + 1):
        elems.append(cur)
        cur = fm(cur, prim)
    vals = [e[0] + e[1] * q for e in elems]
    vals = sorted(set(vals))
    if 0 not in vals:
        vals = [0] + vals
    return vals

def _local_search(U, max_iters=8000, deadline=None):
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200
    no_improve = 0
    max_no_improve = 40

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        improved = False
        idxs = list(range(len(U)))
        random.shuffle(idxs)
        for i in idxs:
            if U[i] == 0:
                continue
            old_val = U[i]
            cands = set()
            for off in range(-30, 31):
                nv = old_val + off
                if nv > 0:
                    cands.add(nv)
            for _ in range(8):
                cands.add(random.randint(1, upper_bound))
            for nv in sorted(cands, key=lambda x: abs(x - old_val)):
                if nv == old_val:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
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
            if len(U) < 300:
                for _ in range(30):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
        if not improved and random.random() < 0.15:
            new_U = U[:]
            nc = random.randint(2, min(8, len(new_U) - 1))
            for _ in range(nc):
                idx = random.randint(1, len(new_U) - 1)
                new_U[idx] = random.randint(1, upper_bound)
            if len(set(new_U)) == len(new_U):
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    no_improve = 0
    return sorted(set(best_U)), best_score

def _polish(U, deadline=None):
    U = sorted(set(U))
    n = len(U)
    if n > 50:
        return _local_search(U, max_iters=3000, deadline=deadline)
    best_score = _score_set_exact(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 80
    no_improve = 0
    for it in range(5000):
        if deadline and time.time() > deadline:
            break
        if no_improve >= 30:
            break
        improved = False
        idxs = list(range(1, len(U)))
        random.shuffle(idxs)
        for i in idxs:
            old_val = U[i]
            for off in range(-20, 21):
                nv = old_val + off
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_exact(new_U)
                if s > best_score + 1e-12:
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
            if len(U) < 70:
                for _ in range(20):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 3:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    random.seed(42)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # Bose Sidon sets for various odd primes q
    for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]:
        if time.time() > deadline - 40:
            break
        try:
            U = _bose_sidon(q)
            if U and 0 in U:
                candidates.append(U)
        except Exception:
            pass

    # Random sets with various parameters
    for trial in range(80):
        if time.time() > deadline - 30:
            break
        max_val = random.randint(30, 1500)
        # Ensure we can sample n-1 distinct elements from 1..max_val
        # Population size is max_val. We need n-1 <= max_val.
        # Also n >= 3.
        max_n = min(60, max_val + 1)
        if max_n < 3:
            continue
        n = random.randint(3, max_n)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # Sparse random
    for trial in range(40):
        if time.time() > deadline - 30:
            break
        max_val = random.randint(200, 4000)
        max_n = min(50, max_val + 1)
        if max_n < 5:
            continue
        n = random.randint(5, max_n)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # Evaluate and keep top
    scored = []
    for U in candidates:
        if time.time() > deadline - 20:
            break
        U = sorted(set(U))
        if 0 not in U:
            continue
        s = _score_set(U)
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]

    scored.sort(key=lambda x: -x[0])

    # Local search on best
    if time.time() < deadline - 15:
        best_U, best_score = _local_search(best_U, max_iters=12000, deadline=deadline - 15)

    # Local search on top candidates
    for s, U in scored[:15]:
        if time.time() > deadline - 15:
            break
        if U == best_U:
            continue
        if s > best_score - 0.05:
            iU, is_ = _local_search(U, max_iters=5000, deadline=deadline - 15)
            if is_ > best_score:
                best_score = is_
                best_U = iU

    # Final polish
    if time.time() < deadline - 3:
        best_U, best_score = _polish(best_U, deadline=deadline - 1)

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
