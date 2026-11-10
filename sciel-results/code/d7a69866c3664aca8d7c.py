"""
Low discrepancy signs on all arithmetic progressions in [1, 256].

Structural change:
- Replace dense incidence matrix with sparse position-to-AP mapping.
- Update only affected AP sums in Greedy and SA loops.
- Recalculate global max via np.max on the small (32k) ap_sums array, which is fast.
- Avoids large temporary array allocations per step.
"""
import time
import random
import math
import numpy as np

_N = 256


def _build_ap_structures():
    """
    Build:
    - ap_positions: list of arrays of positions per AP
    - pos_to_aps: list of lists, pos_to_aps[p] contains indices of APs containing p
    """
    n = _N
    ap_positions = []
    pos_to_aps = [[] for _ in range(n)]
    
    ap_index = 0
    for s in range(n):
        for d in range(1, n):
            positions = []
            x = s
            while x < n:
                positions.append(x)
                x += d
            if len(positions) >= 2:
                pos_arr = np.array(positions, dtype=np.int32)
                ap_positions.append(pos_arr)
                for p in positions:
                    pos_to_aps[p].append(ap_index)
                ap_index += 1
                
    return ap_positions, pos_to_aps


def _greedy_init(n, pos_to_aps, rng=None):
    """
    Greedy constructive initialization.
    """
    if rng is None:
        rng = random.Random(42)
    
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(len(pos_to_aps[0]) if n>0 else 0, dtype=np.int32) # Need num_aps
    # Actually, we need num_aps from the structure. Let's pass it or compute.
    # Better to pass num_aps or get from a global/parameter.
    # Let's modify signature to accept num_aps or infer.
    # In the main construct, we have ap_positions. len(ap_positions) is num_aps.
    
    # Re-define greedy to take num_aps
    pass

def construct():
    start_time = time.time()
    total_budget = 55  # Safety margin
    
    rng = random.Random(123)
    
    # Precompute AP structures
    ap_positions, pos_to_aps = _build_ap_structures()
    num_aps = len(ap_positions)
    
    elapsed = time.time() - start_time
    
    best_signs = None
    best_max = float('inf')
    
    # Phase 1: Multiple greedy initializations
    for trial in range(10): # Increase trials if fast
        signs_g = _greedy_init_fast(_N, pos_to_aps, num_aps, rng)
        max_g = int(np.max(np.abs(_compute_ap_sums(signs_g, ap_positions))))
        if max_g < best_max:
            best_max = max_g
            best_signs = signs_g.copy()
            
    # Phase 2: SA from best initialization
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5 and best_signs is not None:
        result_signs, result_max = _simulated_annealing_fast(best_signs, pos_to_aps, num_aps, ap_positions, remaining - 2, rng)
        if result_max < best_max:
            best_max = result_max
            best_signs = np.array(result_signs, dtype=np.int8)
            
    # Phase 3: More SA or restarts if time
    remaining2 = total_budget - (time.time() - start_time)
    if remaining2 > 10:
        for _ in range(3):
            if time.time() - start_time > total_budget - 5:
                break
            rng3 = random.Random(int(time.time()*1000) % 10000)
            signs_g3 = _greedy_init_fast(_N, pos_to_aps, num_aps, rng3)
            result_signs2, result_max2 = _simulated_annealing_fast(signs_g3, pos_to_aps, num_aps, ap_positions, remaining2 - 5, rng3)
            if result_max2 < best_max:
                best_max = result_max2
                best_signs = np.array(result_signs2, dtype=np.int8)
            remaining2 = total_budget - (time.time() - start_time)
            if remaining2 <= 5:
                break

    return [int(x) for x in best_signs]

def _compute_ap_sums(signs, ap_positions):
    sums = np.zeros(len(ap_positions), dtype=np.int32)
    for i, pos in enumerate(ap_positions):
        sums[i] = np.sum(signs[pos])
    return sums

def _greedy_init_fast(n, pos_to_aps, num_aps, rng):
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    order = list(range(n))
    rng.shuffle(order)
    
    for pos in order:
        best_sign = 1
        best_local_max = float('inf')
        
        affected = pos_to_aps[pos]
        
        for s in (1, -1):
            # Calculate new sums for affected APs
            # We need to simulate the update
            # Current ap_sums is global state.
            # We can't modify it directly until we decide.
            # Since affected count is small, we can compute the new max
            # by checking max of (unaffected max) and (affected new values).
            
            # Optimization: Pre-compute max of unaffected?
            # No, just compute full max on a copy? No, copy is 32k.
            # Let's just update a temporary copy of affected sums and find max.
            # But we need max of ALL sums.
            # The max of all sums is max( max(ap_sums[not_affected]), max(ap_sums[affected] + s) )
            
            # This requires knowing max of not_affected.
            # Simplest: Just update ap_sums, calc max, revert if worse.
            # But we are comparing 1 and -1.
            
            # Strategy:
            # 1. Save current max? No, max changes.
            # 2. For each candidate sign s:
            #    - Apply s to ap_sums[affected]
            #    - Calc new_max = np.max(np.abs(ap_sums))
            #    - Revert ap_sums[affected]
            #    - Keep track of best s
            
            old_vals = ap_sums[affected].copy()
            
            # Try +1
            ap_sums[affected] += 1
            max_p1 = int(np.max(np.abs(ap_sums)))
            ap_sums[affected] = old_vals
            
            # Try -1
            ap_sums[affected] -= 1
            max_m1 = int(np.max(np.abs(ap_sums)))
            ap_sums[affected] = old_vals
            
            if max_p1 <= max_m1:
                best_sign = 1
                best_local_max = max_p1
            else:
                best_sign = -1
                best_local_max = max_m1
                
        # Apply best sign
        signs[pos] = best_sign
        ap_sums[affected] += best_sign
        
    return signs

def _simulated_annealing_fast(signs, pos_to_aps, num_aps, ap_positions, max_time, rng):
    n = len(signs)
    signs = np.array(signs, dtype=np.int8)
    
    # Initialize AP_sums
    ap_sums = _compute_ap_sums(signs, ap_positions)
    
    current_max = int(np.max(np.abs(ap_sums)))
    current_max = max(current_max, 1)
    
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # SA parameters
    T_start = 2.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    T = T_start
    
    while time.time() < deadline:
        p = rng.randint(0, n - 1)
        old_sign = signs[p]
        new_sign = -old_sign
        
        affected = pos_to_aps[p]
        old_vals = ap_sums[affected].copy()
        
        # Apply flip
        ap_sums[affected] += 2 * new_sign # Change is -2*old_sign = 2*new_sign
        
        new_max = int(np.max(np.abs(ap_sums)))
        new_max = max(new_max, 1)
        
        delta_max = new_max - current_max
        
        accept = False
        if delta_max <= 0:
            accept = True
        else:
            if T > 0.001:
                prob = math.exp(-delta_max / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            signs[p] = new_sign
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Revert
            ap_sums[affected] = old_vals
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    return best_signs.tolist(), best_max
