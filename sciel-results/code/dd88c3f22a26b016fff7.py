"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import sys

N = 256

def _get_ap_structure():
    """
    Precompute the structure of all arithmetic progressions.
    Returns:
        ap_indices: List of lists, each list contains indices (0-based) of the AP.
        ap_members: For each position i, list of indices of APs that contain i.
    """
    ap_indices = []
    # We need to map position -> list of AP indices containing it
    # Since we don't know the total count of APs beforehand easily without a first pass,
    # we can do two passes or just build a dictionary.
    
    # First pass: collect all APs
    raw_aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            indices = [start - 1 + k * step for k in range(max_len)]
            for length in range(2, max_len + 1):
                raw_aps.append(indices[:length])
                
    ap_indices = raw_aps
    num_aps = len(ap_indices)
    
    # Second pass: build reverse mapping
    # pos_to_aps[i] = list of ap_index where i is a member
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, idxs in enumerate(ap_indices):
        for i in idxs:
            pos_to_aps[i].append(ap_idx)
            
    return ap_indices, pos_to_aps

def _compute_ap_sums(signs, ap_indices):
    """
    Compute the sum of signs for each AP.
    signs: numpy array of shape (N,) with values -1, 1
    ap_indices: list of lists of indices
    Returns: numpy array of shape (num_aps,) with sums
    """
    # This is O(total_length_of_aps). 
    # To speed this up, we can convert ap_indices to a matrix if possible, 
    # but for N=256, the number of APs is roughly sum_{step=1}^{128} sum_{start=1}^{256-step} (256-step)/step 
    # which is approx N log N. 
    # Actually, let's just use a loop for initialization, it's fast enough once.
    sums = np.zeros(len(ap_indices), dtype=np.int8)
    for ap_idx, idxs in enumerate(ap_indices):
        sums[ap_idx] = np.sum(signs[idxs])
    return sums

def _greedy_descent(signs, ap_indices, pos_to_aps, time_limit, rng):
    """
    Perform greedy coordinate descent.
    At each step, find the flip that reduces the max absolute AP sum the most.
    To make this fast, we maintain current AP sums.
    When we flip position p:
    - All APs containing p change sum by +/- 2.
    - We need to check if the new max abs sum is lower.
    
    Optimization:
    Maintain current_sums.
    Maintain current_max_abs.
    For a candidate flip at p:
    - Identify affected APs (A_p).
    - Calculate new_sums for A_p.
    - The new global max is max( max(new_sums[A_p]), max(old_sums[~A_p]) ).
    - This requires knowing max(old_sums[~A_p]).
    
    Efficient approach:
    Keep a sorted list or histogram of abs(sums)? 
    Or simply: 
    1. Calculate current max M.
    2. For each position p, calculate potential new max if flipped.
       - This is expensive if done for all p every time.
       
    Alternative Efficient Approach (Batched Greedy):
    1. Compute all AP sums.
    2. Find the "worst" AP (max abs sum).
    3. Try flipping one element in that worst AP to reduce its sum.
    4. If it reduces the global max, accept.
    5. Repeat.
    
    Even better: 
    Maintain a priority queue or just scan positions.
    Given N=256 is small, scanning all 256 positions per iteration is O(N * K) where K is avg APs per pos.
    K is roughly log(N). So O(N log N) per iteration.
    If we do 1000 iterations, that's 256 * 10 * 1000 = 2.5M ops. Very fast in Python/NumPy mix.
    
    Let's implement:
    1. Init sums.
    2. Loop:
       a. Find current max_abs.
       b. For each pos p in 0..N-1:
          - Determine delta = 2 * signs[p] (if we flip, sum changes by -2*signs[p] for each AP containing p)
          - Actually, new_sum = old_sum - 2 * signs[p]
          - We need to check if flipping p reduces the max.
          - New max for this flip = max( max(abs(new_sums_affected)), max(abs(old_sums_unaffected)) )
          - To get max(abs(old_sums_unaffected)), we can precompute top 2 max abs sums globally.
            If the current global max AP is not affected by p, then max_unaffected = current_max.
            If it is affected, we need the next highest.
            
    Let's simplify: 
    Since N is small, we can just recompute the max for the affected APs and compare with the "rest".
    To get the "rest" max efficiently:
    Keep a sorted array of abs_sums? Updating it is hard.
    
    Let's stick to a simpler heuristic that is very fast:
    1. Compute sums.
    2. Find max_abs_val and the APs achieving it (worst_aps).
    3. For each position p in the union of indices of worst_aps:
       - Try flipping p.
       - Update sums for affected APs.
       - Check if new max < current max.
       - If yes, accept, break.
    4. If no improvement in worst_aps, try random positions or all positions.
    
    Actually, let's just do full scan of all positions but optimize the max check.
    Precompute: 
    - sums (array)
    - abs_sums (array)
    - max_val = max(abs_sums)
    - count_max = number of APs with abs_sums == max_val
    
    When flipping p:
    - affected = pos_to_aps[p]
    - new_sums_affected = sums[affected] - 2 * signs[p]
    - new_abs_affected = abs(new_sums_affected)
    - max_affected = max(new_abs_affected)
    
    - What is the max of the *unaffected* APs?
      If max_val is achieved by an AP in 'affected', we need the second max.
      If max_val is NOT achieved by an AP in 'affected', max_unaffected = max_val.
      
    We can maintain a histogram of abs_sums values.
    Values are integers from 0 to N (256).
    Histogram size 257.
    
    Update histogram on flip:
    - For each ap in affected:
      - old_val = abs(sums[ap])
      - new_val = abs(sums[ap] - 2*signs[p])
      - hist[old_val] -= 1
      - hist[new_val] += 1
      
    Then max_unaffected can be found by scanning hist from top down, ignoring indices in 'affected'?
    No, hist is global. We need max over APs NOT in affected.
    
    This is getting complex. Given N=256, let's just use the "Recompute Max from Scratch for Affected + Check Global Max" logic but optimized.
    
    Actually, the simplest robust fast method:
    1. Keep `sums`.
    2. Keep `abs_sums`.
    3. Keep `max_val`.
    4. For each p:
       - affected = pos_to_aps[p]
       - new_abs_affected = abs(sums[affected] - 2*signs[p])
       - max_new_affected = max(new_abs_affected)
       
       - We need max of abs_sums for indices NOT in affected.
       - If max_val is not in affected (i.e., no ap in affected has abs_sums[ap] == max_val), then max_unaffected = max_val.
       - If max_val IS in affected, we need the next highest value in abs_sums that is not in affected.
       
    To make this O(1) or O(K):
    Maintain a list of (abs_val, ap_index) sorted descending? 
    Or just: 
    If max_val is affected, scan the sorted list of all abs_sums until we find one not in affected.
    
    Let's implement a version that maintains a sorted list of (abs_val, ap_idx).
    Size ~50k. Sorting is O(N log N). Updating is hard.
    
    Alternative: Just do the naive O(N_aps) check for the "unaffected" max only when necessary?
    No, that's what we want to avoid.
    
    Let's try a different tactic: **Simulated Annealing with Vectorized Evaluation**.
    It's often better than greedy for escaping local minima.
    But the prompt suggests SGD/Gradient. 
    However, for discrete ±1, "Gradient" usually means looking at the derivative of the smooth surrogate.
    
    Let's go with **Greedy Descent with Histogram**.
    Histogram of abs_sums.
    To find max_unaffected:
    We know `affected` set.
    We iterate `v` from `max_val` down to 0.
    If `hist[v] > 0`:
       Check if there is an AP with abs_val v NOT in affected.
       This requires knowing *which* APs have value v.
       
    This is too heavy.
    
    Let's step back. N=256 is small.
    The number of APs is ~2500 (estimated: sum_{d=1}^{128} 256/d ~ 256 * ln(128) ~ 1300. Plus lengths. 
    Actually, number of APs of length >= 2:
    For step 1: 255 APs of len 2, 254 of len 3... Total ~ 255*256/2 ~ 32k for step 1?
    No, for step 1, start 1..255. Len 2..256-start.
    Sum_{start=1}^{255} (256-start) = 255*256/2 = 32640.
    For step 2: ~ 256/2 * 128/2 ~ 8k.
    Total APs is likely around 40,000 - 50,000.
    
    50,000 APs.
    Evaluating max for all 50k APs takes ~1ms in NumPy if vectorized.
    If we do 1000 evaluations, that's 1 second.
    We have 60s. So we can do ~50,000 evaluations.
    
    Strategy:
    1. Precompute AP indices as a list of arrays.
    2. Use a "Batch" approach.
    3. Maintain current signs.
    4. Maintain current sums (vector of 50k ints).
    5. Loop:
       - Pick a random position p (or scan all).
       - Calculate new sums for affected APs.
       - Calculate new max.
       - If improved, accept.
    
    To calculate new max fast:
    We have `sums`.
    We update `sums[affected]`.
    We need `max(abs(sums))`.
    `np.max(np.abs(sums))` on 50k elements is very fast (~0.1ms).
    
    So, we can do:
    1. Flip p.
    2. Update `sums[affected] -= 2*signs[p]`.
    3. `new_max = np.max(np.abs(sums))`.
    4. If `new_max < current_max`, accept. Else, revert `sums[affected] += 2*signs[p]`.
    
    This allows us to test flips very fast.
    We can scan ALL 256 positions in one "pass".
    Cost of one pass: 256 flips * (update + max).
    Update is O(K) where K~10-20. Max is O(50k).
    256 * 50k = 12.8M ops. ~0.1s.
    So we can do ~500 passes in 60s.
    Each pass might find 1-10 improvements.
    This is much better than the current program which does ~50 flips total.
    
    Let's implement this "Scan All Positions" Greedy Descent.
    
    """
    signs = np.array(signs, dtype=np.int8)
    sums = _compute_ap_sums(signs, ap_indices)
    current_max = np.max(np.abs(sums))
    
    # Precompute affected lists as numpy arrays for fast indexing
    affected_lists = [np.array(pos_to_aps[i]) for i in range(N)]
    
    deadline = time.time() + time_limit
    no_improve_passes = 0
    max_no_improve = 10 # Stop if 10 full passes yield no improvement
    
    while time.time() < deadline - 1.0:
        improved = False
        # Randomize order to avoid bias
        order = list(range(N))
        rng.shuffle(order)
        
        for p in order:
            if time.time() > deadline - 1.0:
                break
                
            old_sign = signs[p]
            delta = -2 * old_sign # New sum = Old sum + delta
            
            affected = affected_lists[p]
            if len(affected) == 0:
                continue
                
            # Update sums
            sums[affected] += delta
            
            new_max = np.max(np.abs(sums))
            
            if new_max < current_max:
                # Accept
                signs[p] = -old_sign
                current_max = new_max
                improved = True
            else:
                # Revert
                sums[affected] -= delta
                
        if improved:
            no_improve_passes = 0
        else:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                break
                
    return signs.tolist(), current_max

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, keep 5s margin
    
    print("Computing AP structure...", flush=True)
    ap_indices, pos_to_aps = _get_ap_structure()
    print(f"Number of APs: {len(ap_indices)}", flush=True)
    
    rng = random.Random(42)
    
    best_signs = None
    best_val = float('inf')
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse (often good for discrepancy)
    tm = []
    for i in range(N):
        c = bin(i).count('1')
        tm.append(1 if c % 2 == 0 else -1)
    candidates.append(tm)
    
    # 2. Random
    for i in range(10):
        candidates.append([rng.choice((-1, 1)) for _ in range(N)])
        
    # 3. Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(N)])
    candidates.append([-1 if i % 2 == 0 else 1 for i in range(N)])
    
    # Evaluate and select best initial
    for cand in candidates:
        signs_np = np.array(cand, dtype=np.int8)
        sums = _compute_ap_sums(signs_np, ap_indices)
        val = np.max(np.abs(sums))
        if val < best_val:
            best_val = val
            best_signs = cand[:]
            
    print(f"Best initial: {best_val}", flush=True)
    
    # Greedy Descent
    if time.time() < deadline - 5.0:
        hc_time = min(70.0, (deadline - time.time()) * 0.9)
        best_signs, best_val = _greedy_descent(best_signs, ap_indices, pos_to_aps, hc_time, rng)
        print(f"After greedy descent: {best_val}", flush=True)
        
    # If time remains, try from other good candidates?
    # For now, just return the best found.
    # The greedy descent is robust.
    
    final_val = best_val
    print(f"Final value: {final_val}", flush=True)
    
    return best_signs

# EVOLVE-BLOCK-END
