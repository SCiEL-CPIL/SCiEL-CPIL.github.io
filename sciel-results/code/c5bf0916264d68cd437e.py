"""FFT-accelerated score + structured search over sumset/difference-set exponent."""
import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 6.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()


def _fft_score(U):
    """Score via FFT convolution. U must be nonnegative ints, 0 in U, len>=2."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    # Sumset: conv of indicator with itself, support [0, 2M]
    N = 2 * (2 * M + 1)  # power-of-two-ish; use next pow2 for speed
    # Actually use exact size then next pow2
    size_sum = 2 * M + 1
    N = 1
    while N < size_sum:
        N <<= 1
    v = np.zeros(N, dtype=np.float64)
    for x in U:
        v[x] = 1.0
    # Convolution via FFT
    vf = np.fft.rfft(v)
    conv = np.fft.irfft(vf * vf, n=N)
    # Sumset elements are indices 0..2M where conv > 0.5
    sumset_count = int(np.sum(conv[:size_sum] > 0.5))
    # Difference set: support [-(M), M], count via cross-correlation
    # corr[k] = sum_x v[x] * v[x + k] for k in [-M, M]
    # Use FFT: IFFT(FFT(v) * conj(FFT(v))) gives circular autocorrelation
    acf = np.fft.irfft(vf * np.conj(vf), n=N)
    # acf[i] for i in 0..N-1; differences d = i (for i<=M) and d = i-N (for i>M)
    # Count distinct d in [-M, M] where acf > 0.5
    # acf[0]..acf[M] give d=0..M; acf[N-M]..acf[N-1] give d=-M..-1
    count = int(np.sum(acf[:M + 1] > 0.5)) + int(np.sum(acf[N - M:] > 0.5))
    if sumset_count == 0:
        return 0.0
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(count / sumset_count, 2) / denom


def _score_small(U):
    """Exact O(n^2) score for small sets (n <= ~60)."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom


def _score(U):
    """Dispatch: small sets use exact, large use FFT."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) <= 80:
        return _score_small(U)
    return _fft_score(U)


def _gen_golomb(n):
    """Greedy Golomb ruler: maximize distinct pairwise differences."""
    U = [0]
    for i in range(1, n):
        best = None
        # Search from large to small
        upper = i * i * 3 + 20
        for pos in range(upper, i * i, -1):
            cand = U + [pos]
            diffs = set()
            ok = True
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    d = cand[a] - cand[b]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if ok:
                best = pos
                break
        if best is None:
            best = U[-1] + i * i + 2
        U.append(best)
    return U


def _gen_powers(n, base):
    return [base ** i for i in range(n)]


def _gen_fib(n):
    U = [0]
    if n > 1:
        U.append(1)
    for i in range(2, n):
        U.append(U[-1] + U[-2])
    return U


def _gen_random_structured(n, seed):
    rng = random.Random(seed)
    U = [0]
    cur = 0
    for i in range(1, n):
        gap = rng.randint(1, i * i + 3)
        cur += gap
        U.append(cur)
    return U


def _local_search(U, time_budget=30.0, max_iters=2000):
    """Local search with FFT scoring. Tries add/remove/move."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score(best_U)
    start = time.time()
    rng = random.Random(12345)

    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        if _remaining() < 12.0:
            break
        improved = False
        n = len(best_U)
        M = max(best_U)

        # --- Try adding a candidate ---
        candidates = []
        # Near existing
        for x in best_U:
            for d in range(1, 12):
                candidates.append(x + d)
                if x - d >= 0:
                    candidates.append(x - d)
        # Beyond max
        for d in range(1, 60):
            candidates.append(M + d)
        # Random in range
        for _ in range(40):
            candidates.append(rng.randint(0, int(M * 1.5) + 5))

        for c in candidates:
            if c in best_U:
                continue
            new_U = sorted(best_U + [c])
            s = _score(new_U)
            if s > best_s + 1e-10:
                best_s = s
                best_U = new_U
                improved = True
                break

        if improved:
            continue

        # --- Try removing ---
        if n > 2:
            for i in range(n):
                if best_U[i] == 0:
                    continue
                new_U = best_U[:i] + best_U[i + 1:]
                s = _score(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        if improved:
            continue

        # --- Try moving ---
        for i in range(n):
            if best_U[i] == 0:
                continue
            orig = best_U[i]
            for d in range(1, 12):
                for nv in [orig + d, max(0, orig - d)]:
                    if nv in best_U:
                        continue
                    new_U = list(best_U)
                    new_U[i] = nv
                    new_U = sorted(new_U)
                    s = _score(new_U)
                    if s > best_s + 1e-10:
                        best_s = s
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            # Random perturbation
            if rng.random() < 0.4:
                new_U = list(best_U)
                for _ in range(rng.randint(1, 3)):
                    idx = rng.choice([i for i in range(len(new_U)) if new_U[i] != 0])
                    new_U[idx] = max(0, new_U[idx] + rng.randint(-30, 30))
                    new_U = sorted(set(new_U))
                if 0 not in new_U:
                    new_U.append(0)
                    new_U.sort()
                s = _score(new_U)
                if s > best_s:
                    best_s = s
                    best_U = new_U

    return best_U, best_s


def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    # Generate diverse initial sets
    inits = []
    for n in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]:
        inits.append(_gen_golomb(n))
    for base in [2, 3, 4]:
        for n in [5, 7, 9, 11, 13]:
            inits.append(_gen_powers(n, base))
    for n in [5, 7, 9, 11, 13, 15]:
        inits.append(_gen_fib(n))
    for seed in range(30):
        n = random.randint(4, 25)
        inits.append(_gen_random_structured(n, seed))
    inits.append([0, 1, 3])
    inits.append([0, 1, 4, 10])
    inits.append([0, 1, 3, 7, 12])
    inits.append([0, 2, 5, 11, 23])

    # Deduplicate
    seen = set()
    unique = []
    for s in inits:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique.append(list(s))

    # Score all and sort
    scored = [(_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Multi-start local search on top candidates
    top = [s for _, s in scored[:20]]
    total_budget = 95.0
    per_set = total_budget / len(top)

    for s in top:
        if _remaining() < 15.0:
            break
        alloc = min(per_set, _remaining() - 15.0)
        if alloc < 3.0:
            break
        result, rs = _local_search(s, time_budget=alloc, max_iters=3000)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish
    if _remaining() > 20.0:
        polished, ps = _local_search(best_U, time_budget=min(35.0, _remaining() - 10.0), max_iters=5000)
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)

    return best_U
