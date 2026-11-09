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
    """
    For each position p (0-indexed), store list of tuples (i0, d) 
    where i0 is the start index (0-indexed) of the 4-AP a, a+d, a+2d, a+3d
    and p is one of the four terms.
    """
    affected = [[] for _ in range(n)]
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a-1
            # The 4 terms are i0, i0+d, i0+2d, i0+3d
            for j in range(4):
                p = i0 + j*d
                if p < n:
                    affected[p].append((i0, d))
    return affected

def delta_mono(c, p, affected):
    """
    Compute change in mono-4AP count if we flip position p in numpy array c.
    c is a numpy array of int8.
    """
    old = c[p]
    new = 1 - old
    delta = 0
    if not affected[p]:
        return 0
    
    # Extract all APs containing p
    # i0s: start indices, ds: common differences
    i0s = np.array([x[0] for x in affected[p]], dtype=np.int32)
    ds = np.array([x[1] for x in affected[p]], dtype=np.int32)
    
    # The 4 positions for each AP are:
    # pos0 = i0
    # pos1 = i0 + d
    # pos2 = i0 + 2d
    # pos3 = i0 + 3d
    
    pos0 = i0s
    pos1 = i0s + ds
    pos2 = i0s + 2*ds
    pos3 = i0s + 3*ds
    
    # Get current colors at these positions
    c0 = c[pos0]
    c1 = c[pos1]
    c2 = c[pos2]
    c3 = c[pos3]
    
    # Check if currently monochromatic
    # Mono if c0==c1==c2==c3
    was_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # Determine which position in the AP is p
    # p is at i0 + j*d for j in 0..3
    # We can compute j for each AP entry
    # j = (p - i0) / d
    # Since p is in the AP, this is an integer.
    js = (p - i0s) // ds
    
    # Construct the "new" colors array for the 4 positions
    # If the position is p, use 'new', else use current color
    # We can do this by creating a copy or using np.where on the flattened indices?
    # Actually, it's easier to just create 4 new arrays c0_new, c1_new, etc.
    
    c0_new = np.where(js == 0, new, c0)
    c1_new = np.where(js == 1, new, c1)
    c2_new = np.where(js == 2, new, c2)
    c3_new = np.where(js == 3, new, c3)
    
    # Check if monochromatic after flip
    is_mono_now = (c0_new == c1_new) & (c1_new == c2_new) & (c2_new == c3_new)
    
    # Delta = (# now mono) - (# was mono)
    # We only care about the change
    # Count how many were mono and are not now
    # Count how many were not mono and are now
    # delta = sum(is_mono_now) - sum(was_mono)
    
    # Note: An AP cannot be both was_mono and is_mono_now if it contains p and we flip p?
    # If it was mono, all 4 were 'old'. After flip, p is 'new'. 
    # If old != new (which is true), then the AP is no longer mono (since p is different from others).
    # So was_mono implies not is_mono_now.
    # If it was not mono, it might become mono if the other 3 were 'new' and p was 'old'.
    
    # So delta = count(is_mono_now) - count(was_mono)
    delta = int(np.sum(is_mono_now) - np.sum(was_mono))
    return delta

def local_search(colors, deadline, affected):
    c = np.array(colors, dtype=np.int8)
    cur = count_mono_4ap(c)
    improved = True
    # Use random order to break symmetry and avoid bias
    while improved and time.time() < deadline:
        improved = False
        order = list(range(256))
        random.shuffle(order)
        for p in order:
            if time.time() > deadline:
                return c.tolist(), cur
            d = delta_mono(c, p, affected)
            if d < 0:
                c[p] = 1 - c[p]
                cur += d
                improved = True
    return c.tolist(), cur

def simulated_annealing(colors, deadline, affected, start_temp=5.0, end_temp=0.1):
    c = np.array(colors, dtype=np.int8)
    cur = count_mono_4ap(c)
    best = c.copy()
    best_count = cur
    
    elapsed_budget = deadline - time.time()
    if elapsed_budget <= 0:
        return c.tolist(), cur
    
    # Estimate time per step with a small trial
    t0 = time.time()
    trial_steps = 50
    for i in range(trial_steps):
        p = random.randint(0, 255)
        d = delta_mono(c, p, affected)
        temp = start_temp
        if d < 0 or random.random() < math.exp(-d / max(temp, 1e-9)):
            c[p] = 1 - c[p]
            cur += d
    t1 = time.time()
    time_per_step = (t1 - t0) / trial_steps
    
    # Leave 5 seconds buffer for safety
    max_steps = int((elapsed_budget - 5) / max(time_per_step, 1e-9))
    if max_steps < 10:
        return best.tolist(), best_count
    
    cooling_rate = (end_temp / start_temp) ** (1.0 / max_steps)
    temp = start_temp
    
    for i in range(max_steps):
        if time.time() > deadline - 5:
            break
        p = random.randint(0, 255)
        d = delta_mono(c, p, affected)
        if d < 0 or random.random() < math.exp(-d / max(temp, 1e-9)):
            c[p] = 1 - c[p]
            cur += d
            if cur < best_count:
                best_count = cur
                best = c.copy()
        temp *= cooling_rate
        
    return best.tolist(), best_count

def construct():
    start = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    deadline = start + 140
    
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
    
    # Shifted Thue-Morse variants
    tm2 = thue_morse(260)
    candidates.append(tm2[2:258])
    candidates.append(tm2[4:260])
    
    # Complement of Thue-Morse
    candidates.append([1-x for x in thue_morse(256)])
    
    # XOR with higher bit patterns
    tm_base = thue_morse(256)
    candidates.append([(tm_base[i]) ^ ((i>>2)&1) for i in range(256)])
    candidates.append([(tm_base[i]) ^ ((i>>3)&1) for i in range(256)])
    candidates.append([(tm_base[i]) ^ ((i>>4)&1) for i in range(256)])
    
    # Add some random initializations for diversity
    for _ in range(5):
        candidates.append([random.randint(0,1) for _ in range(256)])
        
    best = None
    best_count = float('inf')
    
    # Score candidates
    scored_cands = []
    for cand in candidates:
        if time.time() > deadline - 60:
            break
        cnt = count_mono_4ap(cand)
        scored_cands.append((cnt, cand))
        
    if not scored_cands:
        best = list(candidates[0])
        best_count = count_mono_4ap(best)
    else:
        scored_cands.sort(key=lambda x: x[0])
        best_count, best = scored_cands[0]
        
    affected = build_affected()
    
    # Run local search on top 3 candidates
    # Allocate time for local search: ~40s
    ls_deadline = deadline - 40
    for i, (cnt, cand) in enumerate(scored_cands[:3]):
        if time.time() > ls_deadline:
            break
        # Give each local search run a portion of the time
        # Simple approach: run until deadline for the best, but we want to try others.
        # Let's give the first one 20s, second 10s, third 10s.
        alloc = 20 if i == 0 else 10
        sub_deadline = time.time() + alloc
        if sub_deadline > ls_deadline:
            sub_deadline = ls_deadline
            
        ls_best, ls_count = local_search(cand, sub_deadline, affected)
        if ls_count < best_count:
            best_count = ls_count
            best = ls_best
            
    # Simulated Annealing
    # Allocate ~30s for SA
    sa_deadline = deadline - 10
    if time.time() < sa_deadline:
        sa_best, sa_count = simulated_annealing(best, sa_deadline, affected)
        if sa_count < best_count:
            best_count = sa_count
            best = sa_best
            
    # Final Local Search Polish
    # Allocate ~5s
    if time.time() < deadline - 2:
        final_deadline = deadline - 2
        final_best, final_count = local_search(best, final_deadline, affected)
        if final_count < best_count:
            best_count = final_count
            best = final_best
            
    return best

