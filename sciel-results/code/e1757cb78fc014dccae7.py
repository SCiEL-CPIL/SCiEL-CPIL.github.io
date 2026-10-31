"""Rewritten program: sparse + local sign-flip optimization for C3' minimization."""
import numpy as np
import time

N = 400

def _gen_msequence(n):
    """Generate a maximum-length sequence of length n (m-sequence)."""
    degree = 7
    period = (1 << degree) - 1
    reg = 1
    seq = []
    for _ in range(period):
        seq.append(reg & 1)
        fb = ((reg >> 0) & 1) ^ ((reg >> 6) & 1)
        reg = (reg >> 1) | (fb << 6)
    mseq = np.array([1.0 if x == 0 else -1.0 for x in seq])
    while len(mseq) < n:
        mseq = np.concatenate([mseq, mseq[:n - len(mseq)]])
    return mseq[:n]

def _init_autoconv(a):
    return np.correlate(a, a, mode='full')

def _update_autoconv_flip(ac, a_old_i, i, n, a):
    """Update autoconv after flipping a[i] from a_old_i to -a_old_i.
    a has the NEW value at i, but contrib only uses a[p] for p != i."""
    k = np.arange(2 * n - 1)
    p1 = i + k - (n - 1)
    p2 = i - k + (n - 1)
    valid1 = (p1 >= 0) & (p1 < n) & (p1 != i)
    valid2 = (p2 >= 0) & (p2 < n) & (p2 != i)
    contrib1 = np.zeros(2 * n - 1)
    contrib1[valid1] = -2.0 * a_old_i * a[p1[valid1]]
    contrib2 = np.zeros(2 * n - 1)
    contrib2[valid2] = -2.0 * a[p2[valid2]] * a_old_i
    return ac + contrib1 + contrib2

def _score(a, ac=None):
    s = np.sum(a)
    if s == 0:
        return np.inf
    if ac is None:
        ac = np.correlate(a, a, mode='full')
    return np.max(np.abs(ac)) / (s * s)

def _local_sign_flip_optimize_fast(a, max_iters=200, deadline=None):
    a = a.copy().astype(np.float64)
    n = len(a)
    ac = _init_autoconv(a)
    best_obj = _score(a, ac)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            old_val = a[i]
            a[i] = -old_val
            ac_new = _update_autoconv_flip(ac, old_val, i, n, a)
            new_obj = _score(a, ac_new)
            if new_obj < best_obj:
                best_obj = new_obj
                ac = ac_new
                improved = True
            else:
                a[i] = old_val
        if not improved:
            break
    return a, best_obj

def _pair_flip_optimize_fast(a, max_iters=50, deadline=None):
    a = a.copy().astype(np.float64)
    n = len(a)
    ac = _init_autoconv(a)
    best_obj = _score(a, ac)
    rng = np.random.default_rng(123)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for _ in range(50):
            if deadline is not None and time.time() > deadline:
                break
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            old_i = a[i]
            old_j = a[j]
            a[i] = -old_i
            a[j] = -old_j
            ac_temp = _update_autoconv_flip(ac, old_i, i, n, a)
            ac_new = _update_autoconv_flip(ac_temp, old_j, j, n, a)
            new_obj = _score(a, ac_new)
            if new_obj < best_obj:
                best_obj = new_obj
                ac = ac_new
                improved = True
            else:
                a[i] = old_i
                a[j] = old_j
        if not improved:
            break
    return a, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140
    best_a = None
    best_obj = np.inf
    rng = np.random.default_rng(42)
    
    a_mseq = _gen_msequence(N)
    obj_mseq = _score(a_mseq)
    if obj_mseq < best_obj:
        best_obj = obj_mseq
        best_a = a_mseq.copy()
    
    a1 = np.ones(N)
    obj1 = _score(a1)
    if obj1 < best_obj:
        best_obj = obj1
        best_a = a1.copy()
    
    for trial in range(30):
        if time.time() > deadline:
            break
        a = rng.choice([-1.0, 1.0], size=N)
        obj = _score(a)
        if obj < best_obj:
            best_obj = obj
            best_a = a.copy()
    
    for period in [N // 2, N // 3, N // 4, N // 5]:
        if time.time() > deadline:
            break
        t = np.arange(N)
        a_chirp = np.sign(np.cos(2 * np.pi * t / period))
        a_chirp[a_chirp == 0] = 1.0
        obj_chirp = _score(a_chirp)
        if obj_chirp < best_obj:
            best_obj = obj_chirp
            best_a = a_chirp.copy()
    
    for num_blocks in [4, 8, 16]:
        if time.time() > deadline:
            break
        block_size = N // num_blocks
        a_block = np.zeros(N)
        for b in range(num_blocks):
            a_block[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -1.0
        obj_block = _score(a_block)
        if obj_block < best_obj:
            best_obj = obj_block
            best_a = a_block.copy()
    
    for density in [0.1, 0.2, 0.3, 0.5]:
        for trial in range(10):
            if time.time() > deadline:
                break
            mask = rng.random(N) < density
            a = np.zeros(N)
            a[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            obj = _score(a)
            if obj < best_obj:
                best_obj = obj
                best_a = a.copy()
    
    candidates = [best_a.copy(), _gen_msequence(N).copy()]
    for _ in range(3):
        if time.time() > deadline:
            break
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    t = np.arange(N)
    a_chirp2 = np.sign(np.cos(2 * np.pi * t / (N // 3)))
    a_chirp2[a_chirp2 == 0] = 1.0
    candidates.append(a_chirp2)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        opt_a, opt_obj = _local_sign_flip_optimize_fast(cand, max_iters=50, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    if time.time() < deadline:
        opt_a, opt_obj = _pair_flip_optimize_fast(best_a, max_iters=30, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    if time.time() < deadline:
        opt_a, opt_obj = _local_sign_flip_optimize_fast(best_a, max_iters=100, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    return best_a
# EVOLVE-BLOCK-END
