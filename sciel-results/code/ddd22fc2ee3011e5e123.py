import time
import numpy as np
import random

def _build_ap_table(n):
    """
    Build a table of all arithmetic progressions within [0, n-1].
    For each AP, store:
      - indices: list of positions in the AP
      - For each position in the AP, store which APs contain it and at what position in that AP
    
    Returns:
      ap_list: list of lists of indices (each inner list is an AP)
      pos_to_aps: list of lists, where pos_to_aps[i] = list of (ap_idx, pos_in_ap) 
                  for all APs containing position i
    """
    ap_list = []
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            indices = list(range(r, n, d))
            if len(indices) == 0:
                continue
            ap_idx = len(ap_list)
            ap_list.append(indices)
            for pos_in_ap, idx in enumerate(indices):
                pos_to_aps[idx].append((ap_idx, pos_in_ap))
    
    return ap_list, pos_to_aps

def _compute_all_ap_sums(signs, ap_list):
    """
    Compute the signed sum for every AP in ap_list.
    Returns an array of length len(ap_list) with the sum for each AP.
    """
    n_aps = len(ap_list)
    sums = np.zeros(n_aps, dtype=np.int64)
    for ap_idx, indices in enumerate(ap_list):
        s = 0
        for idx in indices:
            s += signs[idx]
        sums[ap_idx] = s
    return sums

def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over any arithmetic progression.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
                
    return max_disc

def _find_worst_ap(signs, ap_list):
    """
    Returns (max_disc, ap_idx) for one of the APs achieving max_disc.
    """
    sums = _compute_all_ap_sums(signs, ap_list)
    abs_sums = np.abs(sums)
    max_disc = int(np.max(abs_sums))
    best_ap_idx = int(np.argmax(abs_sums))
    return max_disc, best_ap_idx

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = []
    for i in range(n):
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq.append(1 if c % 2 == 0 else -1)
    return seq

def _greedy_construct(n, rng):
    """
    Construct signs greedily. At each step, pick the sign that minimizes the 
    maximum discrepancy of all APs ending at the current position.
    """
    signs = [0] * n
    
    for i in range(n):
        best_sign = 1
        best_disc = 10**9
        
        for s in [1, -1]:
            signs[i] = s
            # Check all APs ending at i
            max_d = i + 1
            current_max = 0
            for d in range(1, max_d + 1):
                s_val = 0
                j = i
                while j >= 0:
                    s_val += signs[j]
                    j -= d
                    if abs(s_val) > current_max:
                        current_max = abs(s_val)
                if current_max > best_disc:
                    break
            
            if current_max < best_disc:
                best_disc = current_max
                best_sign = s
        
        signs[i] = best_sign
        
    return signs

def construct():
    n = 256
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # Build AP table once
    ap_list, pos_to_aps = _build_ap_table(n)
    n_aps = len(ap_list)
    
    # Start with Greedy Construction
    signs = _greedy_construct(n, None)
    
    # Compute initial AP sums
    ap_sums = _compute_all_ap_sums(signs, ap_list)
    current_disc = int(np.max(np.abs(ap_sums)))
    best_disc = current_disc
    best_signs = signs.copy()
    best_ap_sums = ap_sums.copy()
    
    rng = random.Random(42)
    
    # Local Search with hill climbing
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < deadline:
        # Find the worst AP
        abs_sums = np.abs(ap_sums)
        max_disc = int(np.max(abs_sums))
        worst_ap_idx = int(np.argmax(abs_sums))
        
        if max_disc < best_disc:
            best_disc = max_disc
            best_signs = signs.copy()
            best_ap_sums = ap_sums.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck, restart from best with random perturbation
        if no_improve_count > max_no_improve:
            signs = best_signs.copy()
            ap_sums = best_ap_sums.copy()
            # Perturb a few random positions
            for _ in range(5):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
                # Update all AP sums containing this position
                for ap_idx, pos_in_ap in pos_to_aps[idx]:
                    ap_sums[ap_idx] += 2 * signs[idx]  # Flipping changes sum by 2*new_sign
            no_improve_count = 0
            continue
            
        # The worst AP
        worst_ap = ap_list[worst_ap_idx]
        current_sum = int(ap_sums[worst_ap_idx])
        
        # Find candidates in this AP that have the same sign as current_sum
        target_sign = 1 if current_sum > 0 else -1
        candidates = [idx for idx in worst_ap if signs[idx] == target_sign]
        
        if not candidates:
            no_improve_count += 1
            continue
        
        # Try flipping a random candidate
        idx = rng.choice(candidates)
        old_sign = signs[idx]
        signs[idx] = -signs[idx]
        
        # Update all AP sums containing this position
        delta = 2 * signs[idx]  # Change in sum when flipping
        for ap_idx, pos_in_ap in pos_to_aps[idx]:
            ap_sums[ap_idx] += delta
        
        new_max_disc = int(np.max(np.abs(ap_sums)))
        if new_max_disc > max_disc:
            # Revert
            signs[idx] = old_sign
            for ap_idx, pos_in_ap in pos_to_aps[idx]:
                ap_sums[ap_idx] -= delta
            no_improve_count += 1
        else:
            # Keep the flip
            no_improve_count = 0

    return best_signs
