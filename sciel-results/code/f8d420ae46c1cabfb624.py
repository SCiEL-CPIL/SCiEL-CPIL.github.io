import time
import random
import math
import numpy as np

def count_mono_4ap(colors):
    n = 256
    c = np.array(colors, dtype=np.int8)
    count = 0
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i = a-1
            col = c[i]
            if col == c[i+d] and c[i+d] == c[i+2*d] and c[i+2*d] == c[i+3*d]:
                count += 1
    return count

def thue_morse(n):
    seq = [0]*n
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def build_affected(n=256):
    """For each position p (0-indexed), store list of (d, a) such that p is in the 4-AP a, a+d, a+2d, a+3d"""
    affected = [[] for _ in range(n)]
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a-1
            for j in range(4):
                p = i0 + j*d
                if p < n:
                    affected[p].append((d, a))
    return affected

def delta_mono(c, p, affected):
    """Compute change in mono-4AP count if we flip position p"""
    old = c[p]
    new = 1 - old
    delta = 0
    for (d, a) in affected[p]:
        i0 = a - 1
        # Check if this 4-AP is currently monochromatic
        was_mono = (c[i0] == c[i0+d] and c[i0+d] == c[i0+2*d] and c[i0+2*d] == c[i0+3*d])
        # After flipping p, check if it becomes monochromatic
        # p is at position i0 + j*d for some j in 0..3
        # We need to check if all four positions have the same color after flip
        # The other three positions must all equal new for the AP to be mono after flip
        other_vals = []
        for j in range(4):
            pos = i0 + j*d
            if pos != p:
                other_vals.append(c[pos])
        is_mono_now = (len(other_vals) == 3 and other_vals[0] == other_vals[1] == other_vals[2] == new)
        if is_mono_now and not was_mono:
            delta += 1
        elif was_mono and not is_mono_now:
            delta -= 1
    return delta

def local_search(colors, deadline, affected):
    c = list(colors)
    cur = count_mono_4ap(c)
    improved = True
    # Use random order to break symmetry
    while improved and time.time() < deadline:
        improved = False
        order = list(range(256))
        random.shuffle(order)
        for p in order:
            if time.time() > deadline:
                return c, cur
            d = delta_mono(c, p, affected)
            if d < 0:
                c[p] = 1 - c[p]
                cur += d
                improved = True
    return c, cur

def simulated_annealing(colors, deadline, affected, start_temp=5.0, end_temp=0.1):
    c = list(colors)
    cur = count_mono_4ap(c)
    best = list(c)
    best_count = cur
    elapsed_budget = deadline - time.time()
    if elapsed_budget <= 0:
        return c, cur
    # Estimate time per step with a small trial
    t0 = time.time()
    trial_steps = 30
    for i in range(trial_steps):
        p = random.randint(0, 255)
        d = delta_mono(c, p, affected)
        temp = start_temp
        if d < 0 or random.random() < math.exp(-d / max(temp, 1e-9)):
            c[p] = 1 - c[p]
            cur += d
    t1 = time.time()
    time_per_step = (t1 - t0) / trial_steps
    max_steps = int(elapsed_budget / max(time_per_step, 1e-9))
    if max_steps < 10:
        return best, best_count
    
    # Use geometric cooling for better exploration
    cooling_rate = (end_temp / start_temp) ** (1.0 / max_steps)
    temp = start_temp
    for i in range(max_steps):
        if time.time() > deadline - 2:
            break
        p = random.randint(0, 255)
        d = delta_mono(c, p, affected)
        if d < 0 or random.random() < math.exp(-d / max(temp, 1e-9)):
            c[p] = 1 - c[p]
            cur += d
            if cur < best_count:
                best_count = cur
                best = list(c)
        temp *= cooling_rate
    return best, best_count

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    candidates = []
    candidates.append(thue_morse(256))
    candidates.append([(i//2)%2 for i in range(256)])
    candidates.append([(i//3)%2 for i in range(256)])
    candidates.append([(i//4)%2 for i in range(256)])
    tm_full = thue_morse(257)
    candidates.append(tm_full[1:257])
    def v2_color(n):
        out = []
        for i in range(1, n+1):
            x = i
            p = 0
            while x % 2 == 0:
                x //= 2
                p += 1
            out.append(p % 2)
        return out
    candidates.append(v2_color(256))
    candidates.append([(i % 2) ^ ((i//2) % 2) for i in range(256)])
    # Add shifted Thue-Morse variants
    tm2 = thue_morse(260)
    candidates.append(tm2[2:258])
    candidates.append(tm2[4:260])
    # Add complement of Thue-Morse
    candidates.append([1-x for x in thue_morse(256)])
    # Add XOR with higher bit patterns
    candidates.append([(thue_morse(256)[i]) ^ ((i>>2)&1) for i in range(256)])
    candidates.append([(thue_morse(256)[i]) ^ ((i>>3)&1) for i in range(256)])
    candidates.append([(thue_morse(256)[i]) ^ ((i>>4)&1) for i in range(256)])
    # Add random initializations for diversity
    for _ in range(5):
        candidates.append([random.randint(0,1) for _ in range(256)])
    
    best = None
    best_count = float('inf')
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        cnt = count_mono_4ap(cand)
        if cnt < best_count:
            best_count = cnt
            best = list(cand)
    
    if best is None:
        best = list(candidates[0])
        best_count = count_mono_4ap(best)
    
    affected = build_affected()
    
    # Multi-start: run local search on top 3 candidates
    scored_cands = []
    for cand in candidates[:10]:
        if time.time() > deadline - 40:
            break
        cnt = count_mono_4ap(cand)
        scored_cands.append((cnt, cand))
    scored_cands.sort(key=lambda x: x[0])
    
    # Run local search on best few starts
    for i, (cnt, cand) in enumerate(scored_cands[:3]):
        if time.time() > deadline - 25:
            break
        ls_best, ls_count = local_search(cand, deadline - 20 - i*5, affected)
        if ls_count < best_count:
            best_count = ls_count
            best = ls_best
    
    # Simulated annealing to escape local minima
    if time.time() < deadline - 15:
        sa_best, sa_count = simulated_annealing(best, deadline - 8, affected)
        if sa_count < best_count:
            best_count = sa_count
            best = sa_best
    
    # Final greedy polish
    if time.time() < deadline - 5:
        best, best_count = local_search(best, deadline - 2, affected)
    
    return best
