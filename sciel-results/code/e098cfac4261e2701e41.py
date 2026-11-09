import time
import random
import math
import numpy as np

def count_mono_4ap(colors):
    """
    Count monochromatic 4-APs using vectorized operations.
    colors: numpy array of shape (256,) with values 0 or 1.
    """
    n = 256
    c = colors
    count = 0
    # Precomputed list of APs as indices
    # We can just iterate d and a, but vectorized is better if we have the array
    # Let's use the global AP_INDICES if available, or compute on fly
    # For standalone function, we compute on fly or assume global
    if 'AP_INDICES' in globals():
        aps = AP_INDICES
        # c[aps] gives shape (N, 4)
        # Check if all 4 are equal
        # c[aps][:, 0] == c[aps][:, 1] == c[aps][:, 2] == c[aps][:, 3]
        mono = (c[aps][:, 0] == c[aps][:, 1]) & \
               (c[aps][:, 1] == c[aps][:, 2]) & \
               (c[aps][:, 2] == c[aps][:, 3])
        count = np.sum(mono)
    else:
        # Fallback
        for d in range(1, 86):
            max_a = n - 3*d
            for a in range(1, max_a+1):
                i = a-1
                if c[i] == c[i+d] == c[i+2*d] == c[i+3*d]:
                    count += 1
    return count

def thue_morse(n):
    seq = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def build_affected_and_others(n=256):
    """
    For each position p, store the indices of the other 3 elements in each 4-AP containing p.
    Returns a list of numpy arrays `others[p]` of shape (num_aps_containing_p, 3).
    """
    others = [None] * n
    # First, collect all APs
    # We need to map position p to the APs it is in.
    # Let's create a list for each p of tuples (ap_id, pos_in_ap)
    # Or just directly build the 'others' matrix.
    
    # To do this efficiently, we first list all APs.
    ap_list = []
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a - 1
            ap_list.append((i0, i0+d, i0+2*d, i0+3*d))
    
    # Initialize lists for each position
    temp_others = [[] for _ in range(n)]
    
    for ap in ap_list:
        i0, i1, i2, i3 = ap
        # p = i0, others are i1, i2, i3
        temp_others[i0].append([i1, i2, i3])
        # p = i1, others are i0, i2, i3
        temp_others[i1].append([i0, i2, i3])
        # p = i2, others are i0, i1, i3
        temp_others[i2].append([i0, i1, i3])
        # p = i3, others are i0, i1, i2
        temp_others[i3].append([i0, i1, i2])
        
    # Convert to numpy arrays
    for p in range(n):
        if temp_others[p]:
            others[p] = np.array(temp_others[p], dtype=np.int32)
        else:
            others[p] = np.empty((0, 3), dtype=np.int32)
            
    return others, np.array(ap_list, dtype=np.int32)

def delta_mono_fast(c, p, others):
    """
    Calculate delta in mono count if we flip position p.
    c: current colors (numpy array)
    p: position to flip
    others: precomputed list of 'others' arrays
    """
    old_val = c[p]
    o = others[p]
    if len(o) == 0:
        return 0
    
    # Get the values of the other 3 bits for all APs containing p
    # c[o] has shape (N_aps, 3)
    vals = c[o]
    sums = np.sum(vals, axis=1) # shape (N_aps,)
    
    # Logic:
    # If old_val == 0 (flipping to 1):
    #   Delta = -1 if sum == 0 (others all 0)
    #   Delta = +1 if sum == 3 (others all 1)
    #   Delta = 0 otherwise
    # If old_val == 1 (flipping to 0):
    #   Delta = -1 if sum == 3 (others all 1)
    #   Delta = +1 if sum == 0 (others all 0)
    #   Delta = 0 otherwise
    
    if old_val == 0:
        # -1 where sum==0, +1 where sum==3
        delta = np.where(sums == 0, -1, 0) + np.where(sums == 3, 1, 0)
    else:
        # -1 where sum==3, +1 where sum==0
        delta = np.where(sums == 3, -1, 0) + np.where(sums == 0, 1, 0)
        
    return np.sum(delta)

def local_search_fast(c, deadline, others):
    """
    Greedy local search.
    """
    cur_count = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            d = delta_mono_fast(c, p, others)
            if d < 0:
                c[p] = 1 - c[p]
                cur_count += d
                improved = True
    return c.copy(), cur_count

def simulated_annealing_fast(c_init, deadline, others, start_temp=10.0, end_temp=0.1):
    """
    Simulated Annealing with fast delta.
    """
    c = c_init.copy()
    cur_count = count_mono_4ap(c)
    best_c = c.copy()
    best_count = cur_count
    
    # Estimate step time
    t0 = time.time()
    trial_steps = 100
    for _ in range(trial_steps):
        p = random.randint(0, 255)
        _ = delta_mono_fast(c, p, others)
    t1 = time.time()
    time_per_step = (t1 - t0) / trial_steps
    
    elapsed = time.time() - t0
    budget = deadline - time.time()
    max_steps = int(budget / max(time_per_step, 1e-6))
    
    if max_steps <= 0:
        return best_c, best_count

    # Cooling schedule
    # Linear or exponential? Exponential is standard.
    # T(t) = T_start * (T_end/T_start)^(t/max_steps)
    # Or simple linear interpolation in log space?
    # Let's use linear interpolation for simplicity and speed, or geometric.
    # Geometric cooling: T_i = T_start * alpha^i
    # We want T_max_steps = T_end.
    # alpha = (T_end / T_start) ** (1.0 / max_steps)
    alpha = (end_temp / start_temp) ** (1.0 / max_steps)
    
    temp = start_temp
    for i in range(max_steps):
        if time.time() > deadline - 2:
            break
            
        p = random.randint(0, 255)
        d = delta_mono_fast(c, p, others)
        
        # Acceptance probability
        if d < 0:
            accept = True
        else:
            # Avoid underflow
            if temp < 1e-6:
                accept = False
            else:
                # exp(-d/temp)
                # If d is large and temp small, exp is 0
                ratio = math.exp(-d / temp)
                accept = random.random() < ratio
        
        if accept:
            c[p] = 1 - c[p]
            cur_count += d
            if cur_count < best_count:
                best_count = cur_count
                best_c = c.copy()
                
        temp *= alpha
        
    return best_c, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    n = 256
    
    # Precompute structures
    others, ap_indices = build_affected_and_others(n)
    # Store in global for count_mono_4ap to use if needed, though we pass c
    global AP_INDICES
    AP_INDICES = ap_indices
    
    # Generate candidates
    candidates = []
    
    # 1. Thue-Morse
    tm = thue_morse(n)
    candidates.append(tm)
    
    # 2. Complement of Thue-Morse
    candidates.append(1 - tm)
    
    # 3. Random
    for _ in range(5):
        candidates.append(np.random.randint(0, 2, n, dtype=np.int8))
        
    # 4. Block patterns
    # Period 2
    p2 = np.array([i % 2 for i in range(n)], dtype=np.int8)
    candidates.append(p2)
    # Period 3
    p3 = np.array([i % 3 % 2 for i in range(n)], dtype=np.int8)
    candidates.append(p3)
    # Period 4
    p4 = np.array([i % 4 % 2 for i in range(n)], dtype=np.int8)
    candidates.append(p4)
    
    # Evaluate initial candidates
    best_c = None
    best_count = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        cnt = count_mono_4ap(cand)
        if cnt < best_count:
            best_count = cnt
            best_c = cand.copy()
            
    if best_c is None:
        best_c = candidates[0]
        best_count = count_mono_4ap(best_c)
        
    # Local Search Phase 1
    if time.time() < deadline - 20:
        best_c, best_count = local_search_fast(best_c, deadline - 20, others)
        
    # Simulated Annealing Phase
    if time.time() < deadline - 10:
        sa_best_c, sa_best_count = simulated_annealing_fast(best_c, deadline - 5, others)
        if sa_best_count < best_count:
            best_c, best_count = sa_best_c, sa_best_count
            
    # Local Search Phase 2 (Polish)
    if time.time() < deadline - 2:
        best_c, best_count = local_search_fast(best_c, deadline - 2, others)
        
    # Convert to list of ints for return
    return best_c.tolist()

if __name__ == "__main__":
    # Quick test
    res = construct()
    assert len(res) == 256
    assert all(x in [0, 1] for x in res)
    print("Count:", count_mono_4ap(np.array(res, dtype=np.int8)))
