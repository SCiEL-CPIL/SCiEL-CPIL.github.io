import time
import random
import math
import numpy as np

N = 256

def count_mono_4ap(colors):
    """Count monochromatic 4-APs. colors is a list or array of 256 values in {0,1}."""
    c = np.asarray(colors, dtype=np.int8)
    total = 0
    for d in range(1, 86):
        max_a = N - 3 * d
        # a ranges 1..max_a, indices a-1..a-1+3d
        base = np.arange(max_a)  # a-1 for a in 1..max_a
        i0 = base
        i1 = base + d
        i2 = base + 2 * d
        i3 = base + 3 * d
        mono = (c[i0] == c[i1]) & (c[i1] == c[i2]) & (c[i2] == c[i3])
        total += int(mono.sum())
    return total

# Precompute all 4-AP index tuples
AP_LIST = []
for d in range(1, 86):
    max_a = N - 3 * d
    for a in range(1, max_a + 1):
        i0 = a - 1
        AP_LIST.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
AP_LIST = np.array(AP_LIST, dtype=np.int32)
NUM_APS = len(AP_LIST)

# Precompute for each position, the indices of APs that contain it
POS_AP_IDX = [[] for _ in range(N)]
for ap_idx, (i0, i1, i2, i3) in enumerate(AP_LIST):
    for p in (i0, i1, i2, i3):
        POS_AP_IDX[p].append(ap_idx)

def thue_morse(n):
    seq = [0] * n
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def build_candidates():
    cands = []
    tm = thue_morse(N)
    cands.append(tm)
    cands.append([1 - x for x in tm])
    cands.append([(i // 2) % 2 for i in range(N)])
    cands.append([(i // 3) % 2 for i in range(N)])
    cands.append([(i // 4) % 2 for i in range(N)])
    cands.append([(i // 5) % 2 for i in range(N)])
    cands.append([(i % 2) ^ ((i // 2) % 2) for i in range(N)])
    cands.append([(i % 3) ^ ((i // 3) % 2) for i in range(N)])
    # v2 coloring
    v2 = []
    for i in range(1, N + 1):
        x = i
        p = 0
        while x % 2 == 0:
            x //= 2
            p += 1
        v2.append(p % 2)
    cands.append(v2)
    # Shifted Thue-Morse
    tm_long = thue_morse(N + 8)
    for s in [1, 2, 3, 4, 5, 6, 7]:
        cands.append(tm_long[s:s + N])
    # XOR with higher bits
    for b in [2, 3, 4, 5, 6]:
        cands.append([(tm[i]) ^ ((i >> b) & 1) for i in range(N)])
    # Random candidates
    rng = random.Random(42)
    for _ in range(10):
        cands.append([rng.randint(0, 1) for _ in range(N)])
    return cands

def delta_flip(c, p):
    """Compute change in mono-4AP count if we flip position p."""
    old = c[p]
    new = 1 - old
    delta = 0
    for ap_idx in POS_AP_IDX[p]:
        i0, i1, i2, i3 = AP_LIST[ap_idx]
        # Count how many of the 4 positions have old color vs new color
        # Before flip: all 4 positions have their current values
        # After flip: position p has new value
        # An AP is mono before if all 4 are same
        # An AP is mono after if all 4 are same (with p flipped)
        
        # Get the other 3 positions' values
        others = []
        for idx in (i0, i1, i2, i3):
            if idx != p:
                others.append(c[idx])
        
        # Before flip: p has old, others have their values
        # After flip: p has new, others have their values
        
        # Before: mono if old == all others
        before_mono = (old == others[0]) and (others[0] == others[1]) and (others[1] == others[2])
        # After: mono if new == all others
        after_mono = (new == others[0]) and (others[0] == others[1]) and (others[1] == others[2])
        
        if before_mono and not after_mono:
            delta -= 1
        elif after_mono and not before_mono:
            delta += 1
    
    return delta

def local_search(c, deadline):
    """Greedy local search: flip positions if it reduces mono-4AP count."""
    c = c.copy()
    cur = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(N):
            if time.time() > deadline:
                break
            d = delta_flip(c, p)
            if d < 0:
                c[p] = 1 - c[p]
                cur += d
                improved = True
    return c, cur

def simulated_annealing(c, deadline, t_start=5.0, t_end=0.05):
    """Simulated annealing to escape local minima."""
    c = c.copy()
    cur = count_mono_4ap(c)
    best = c.copy()
    best_count = cur
    
    elapsed = deadline - time.time()
    if elapsed <= 0:
        return best, best_count
    
    # Estimate time per step
    t0 = time.time()
    for _ in range(20):
        p = random.randint(0, N - 1)
        _ = delta_flip(c, p)
    t1 = time.time()
    time_per_step = (t1 - t0) / 20
    max_steps = int(elapsed / max(time_per_step, 1e-9))
    
    for step in range(max_steps):
        if time.time() > deadline - 2:
            break
        p = random.randint(0, N - 1)
        d = delta_flip(c, p)
        temp = t_start * (1 - step / max(max_steps, 1)) + t_end
        if d < 0 or random.random() < math.exp(-d / max(temp, 1e-9)):
            c[p] = 1 - c[p]
            cur += d
            if cur < best_count:
                best_count = cur
                best = c.copy()
    return best, best_count

def spectral_refine(c, deadline):
    """Spectral refinement: use eigenvalue-based approach to guide flips."""
    c = c.copy()
    cur = count_mono_4ap(c)
    best = c.copy()
    best_count = cur
    
    # Try to use the structure of the problem
    # The key insight: positions with many monochromatic APs are "bad"
    # Flipping them is more likely to help
    
    # Compute for each position, how many APs it's in that are currently mono
    mono_count_per_pos = np.zeros(N, dtype=np.int32)
    for ap_idx in range(NUM_APS):
        i0, i1, i2, i3 = AP_LIST[ap_idx]
        if c[i0] == c[i1] == c[i2] == c[i3]:
            for p in (i0, i1, i2, i3):
                mono_count_per_pos[p] += 1
    
    # Prioritize flipping positions in many mono APs
    order = np.argsort(-mono_count_per_pos)
    
    for p in order:
        if time.time() > deadline:
            break
        d = delta_flip(c, p)
        if d < 0:
            c[p] = 1 - c[p]
            cur += d
            if cur < best_count:
                best_count = cur
                best = c.copy()
    
    return best, best_count

def multi_start_local_search(cands, deadline):
    """Run local search from multiple starting points."""
    best = None
    best_count = float('inf')
    
    # First, evaluate all candidates
    scored = []
    for cand in cands:
        cnt = count_mono_4ap(cand)
        scored.append((cnt, cand))
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates and run local search
    max_starts = 5
    time_per_start = (deadline - time.time()) / max_starts
    for cnt, cand in scored[:max_starts]:
        if time.time() > deadline:
            break
        t_end = time.time() + max(time_per_start, 1)
        c, c_count = local_search(np.array(cand, dtype=np.int8), t_end)
        if c_count < best_count:
            best_count = c_count
            best = c
    
    if best is None:
        best = np.array(cands[0], dtype=np.int8)
        best_count = count_mono_4ap(best)
    
    return best, best_count

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    # Generate and evaluate candidates
    cands = build_candidates()
    best, best_count = multi_start_local_search(cands, deadline - 20)
    
    # Spectral refinement
    if time.time() < deadline - 15:
        spec_best, spec_count = spectral_refine(best, deadline - 15)
        if spec_count < best_count:
            best, best_count = spec_best, spec_count
    
    # Simulated annealing
    if time.time() < deadline - 10:
        sa_best, sa_count = simulated_annealing(best, deadline - 5)
        if sa_count < best_count:
            best, best_count = sa_best, sa_count
    
    # Final local search polish
    if time.time() < deadline - 3:
        best, best_count = local_search(best, deadline - 1)
    
    return best.tolist()

