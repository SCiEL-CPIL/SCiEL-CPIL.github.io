# EVOLVE-BLOCK-START
import time
import numpy as np

def _precompute_aps(n=256):
    ap_indices = []
    for d in range(1, n):
        for s in range(n):
            idx = np.arange(s, n, d, dtype=np.int32)
            ap_indices.append(idx)
    return ap_indices

def _max_disc(signs, ap_indices):
    max_abs = 1
    for idx in ap_indices:
        c = np.cumsum(signs[idx])
        m = np.max(np.abs(c))
        if m > max_abs:
            max_abs = m
    return max_abs

def _thue_morse(n):
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x = i
        p = 0
        while x:
            p ^= 1
            x >>= 1
        seq[i] = 1 if p == 0 else -1
    return seq

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80
    
    ap_indices = _precompute_aps(n)
    
    candidates = [
        _thue_morse(n),
        np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8),
        np.array([1 if i % 3 < 2 else -1 for i in range(n)], dtype=np.int8),
        np.array([1 if (i % 4) < 2 else -1 for i in range(n)], dtype=np.int8),
    ]
    rng = np.random.RandomState(42)
    for _ in range(8):
        candidates.append(rng.choice([-1, 1], size=n).astype(np.int8))
    
    best_signs = None
    best_disc = 999
    for cand in candidates:
        d = _max_disc(cand, ap_indices)
        if d < best_disc:
            best_disc = d
            best_signs = cand.copy()
    
    signs = best_signs.copy()
    np_random = np.random.RandomState(7)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        pos = np_random.randint(n)
        old = signs[pos]
        signs[pos] = -old
        new_disc = _max_disc(signs, ap_indices)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = signs.copy()
            no_improve = 0
        else:
            signs[pos] = old
            no_improve += 1
        
        if no_improve >= max_no_improve:
            signs = np_random.choice([-1, 1], size=n).astype(np.int8)
            d = _max_disc(signs, ap_indices)
            if d < best_disc:
                best_disc = d
                best_signs = signs.copy()
            no_improve = 0
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
