import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 115.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _random_bitseq(rng):
    x = rng.random(N)
    idx = np.argsort(x)
    seq = np.zeros(N)
    seq[idx[HALF:]] = 1.0
    return seq

def _sa(rng, start, iters, T0, T1, t_start, deadline):
    h = start.copy()
    cur = _max_overlap(h)
    best = h.copy()
    bestc = cur
    for it in range(iters):
        if time.time() > deadline:
            break
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        h[i], h[j] = h[j], h[i]
        newc = _max_overlap(h)
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
        if newc < cur or rng.random() < np.exp(-(newc - cur) / max(T, 1e-9)):
            cur = newc
            if newc < bestc:
                bestc = newc
                best = h.copy()
        else:
            h[i], h[j] = h[j], h[i]
    return best, bestc

def _local_search(h, t_start, deadline):
    """
    Exhaustive single-swap local search.
    Tries every possible swap of a 1 and a 0.
    """
    h = h.copy()
    cur = _max_overlap(h)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        for i in ones:
            for j in zeros:
                if time.time() > deadline:
                    return h, cur
                orig_i, orig_j = h[i], h[j]
                h[i], h[j] = h[j], h[i]
                newc = _max_overlap(h)
                if newc < cur:
                    cur = newc
                    improved = True
                    # Restart scan? Or continue?
                    # Continuing is faster but might miss global min if landscape is rugged.
                    # Let's break and restart to ensure we find the local minimum properly.
                    break
                else:
                    h[i], h[j] = orig_i, orig_j
            if improved:
                break
    return h, cur

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    bestc = np.inf
    deadline = t0 + DEADLINE

    # Strategy:
    # 1. Generate diverse binary starting points.
    # 2. Run Simulated Annealing on each.
    # 3. Refine the best SA result with exhaustive Local Search.
    # 4. If time remains, try a few more random restarts with LS.

    starts = []
    # Alternating
    alt = np.zeros(N); alt[::2] = 1.0
    starts.append(alt)
    # Block
    blk = np.zeros(N); blk[:HALF] = 1.0
    starts.append(blk)
    # Random
    for _ in range(8):
        starts.append(_random_bitseq(rng))

    # Phase 1: SA on diverse starts
    # Allocate ~60% of time to SA
    sa_deadline = t0 + (DEADLINE * 0.6)
    
    for s in starts:
        if time.time() > sa_deadline:
            break
        # Adaptive iters based on remaining time? 
        # Just use a fixed reasonable number that fits in the budget.
        # N=100, FFT is fast. 20000 iters is probably < 1s.
        h, c = _sa(rng, s, iters=20000, T0=2.0, T1=0.05, t_start=t0, deadline=sa_deadline)
        if c < bestc:
            bestc = c
            best = h

    if best is None:
        best = np.full(N, 0.5)
        bestc = _max_overlap(best)

    # Phase 2: Local Search on the best candidate
    # LS is O(N^2 * CostEval). 2500 swaps. CostEval is ~FFT(100). Very fast.
    # We can run this multiple times if we restart LS from slightly perturbed states?
    # Or just run it on the best SA result.
    
    h_ls, c_ls = _local_search(best, t0, deadline - 1.0)
    if c_ls < bestc:
        bestc = c_ls
        best = h_ls

    # Phase 3: If we have more time, do a few more "SA + LS" cycles with random starts
    # This helps escape the local minimum found by the first LS.
    while time.time() < deadline - 2.0:
        s = _random_bitseq(rng)
        # Shorter SA for these
        h_sa, c_sa = _sa(rng, s, iters=5000, T0=1.0, T1=0.01, t_start=t0, deadline=deadline - 5.0)
        
        # If SA result is promising, try LS
        if c_sa < bestc + 0.1: # Only bother with LS if it's close
             h_ls2, c_ls2 = _local_search(h_sa, t0, deadline - 1.0)
             if c_ls2 < bestc:
                 bestc = c_ls2
                 best = h_ls2

    # Final check: ensure sum is exactly HALF
    # Binary vectors from SA/LS always sum to HALF.
    # But if we ever allowed non-binary, we'd need to normalize.
    # Here we stick to binary, so sum is exactly 50.
    
    return best
