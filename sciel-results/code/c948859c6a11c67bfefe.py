"""
Simulated annealing with Thue-Morse init + vectorized NumPy evaluation.
Structural change: uses temperature-based acceptance to escape local minima,
Thue-Morse sequence as a low-discrepancy starting point, and multi-start
with different seeds to explore diverse basins of attraction.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

_N = 256

def _thue_morse(n):
    """
    Generate Thue-Morse sequence of length n: -1 for even parity, +1 for odd parity.
    The Thue-Morse sequence has known low-discrepancy properties.
    """
    signs = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        # Count bits set in i
        x = i
        parity = 0
        while x:
            parity ^= 1
            x &= x - 1
        signs[i] = 1 if parity else -1
    return signs

def _generate_ap_matrix(n):
    """
    Generate all arithmetic progressions in [0, n-1] with length >= 2.
    Returns (ap_matrix, ap_lengths) where ap_matrix is a 2D numpy array
    with shape (num_aps, max_ap_len) padded with -1, and ap_lengths is
    the actual length of each AP.
    """
    ap_list = []
    for d in range(1, n):
        for r in range(d):
            seq = list(range(r, n, d))
            if len(seq) > 1:
                ap_list.append(seq)
    
    num_aps = len(ap_list)
    if num_aps == 0:
        return np.zeros((0, 0), dtype=np.int32), np.zeros(0, dtype=np.int32)
    
    max_len = max(len(ap) for ap in ap_list)
    ap_matrix = np.full((num_aps, max_len), -1, dtype=np.int32)
    ap_lengths = np.zeros(num_aps, dtype=np.int32)
    
    for i, ap in enumerate(ap_list):
        ap_matrix[i, :len(ap)] = np.array(ap, dtype=np.int32)
        ap_lengths[i] = len(ap)
    
    return ap_matrix, ap_lengths

def _compute_ap_sums(signs, ap_matrix, ap_lengths):
    """
    Compute the sum of signs over each AP using vectorized operations.
    Returns an array of shape (num_aps,) with the signed sum for each AP.
    """
    num_aps = ap_lengths.shape[0]
    sums = np.zeros(num_aps, dtype=np.int32)
    
    for i in range(num_aps):
        length = ap_lengths[i]
        indices = ap_matrix[i, :length]
        sums[i] = np.sum(signs[indices])
    
    return sums

def _compute_max_discrepancy(signs, ap_matrix, ap_lengths):
    """
    Compute the maximum absolute signed sum over all APs.
    """
    sums = _compute_ap_sums(signs, ap_matrix, ap_lengths)
    if len(sums) == 0:
        return 1
    return int(np.max(np.abs(sums)))

def _precompute_ap_containing_fast(ap_matrix, ap_lengths, n):
    """
    Precompute which APs contain each position.
    Returns a list of lists: ap_containing[pos] = list of AP indices containing pos.
    """
    num_aps = ap_lengths.shape[0]
    ap_containing = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        length = ap_lengths[ap_idx]
        for pos in ap_matrix[ap_idx, :length]:
            ap_containing[pos].append(ap_idx)
    return ap_containing

def _simulate_annealing(signs, ap_matrix, ap_lengths, ap_containing, start_time, deadline, rng):
    """
    Simulated annealing: try random flips, accept with probability based on
    temperature and energy change. Temperature decreases over time.
    """
    n = _N
    num_aps = ap_lengths.shape[0]
    
    current_signs = signs.copy()
    current_sums = _compute_ap_sums(current_signs, ap_matrix, ap_lengths)
    current_max = int(np.max(np.abs(current_sums))) if num_aps > 0 else 1
    
    best_signs = current_signs.copy()
    best_max = current_max
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    T_decay = 0.9995  # Exponential decay per iteration
    
    T = T_start
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        old_sign = current_signs[pos]
        new_sign = -old_sign
        
        # Get affected APs from precomputed structure
        affected_aps = ap_containing[pos]
        
        if not affected_aps:
            continue
        
        # Compute old and new max
        old_sums = current_sums[affected_aps].copy()
        
        # Apply flip
        current_signs[pos] = new_sign
        delta = -2 * old_sign
        current_sums[affected_aps] += delta
        
        new_max = int(np.max(np.abs(current_sums)))
        
        # Accept or reject
        if new_max <= current_max:
            # Accept (improvement or tie)
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = current_signs.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Worsening move: accept with probability exp(-delta_E / T)
            delta_E = new_max - current_max
            prob = math.exp(-delta_E / T) if T > 0 else 0
            if rng.random() < prob:
                # Accept worsening move
                current_max = new_max
            else:
                # Reject: revert
                current_signs[pos] = old_sign
                current_sums[affected_aps] = old_sums
        
        # Decay temperature
        T = max(T * T_decay, T_end)
        
        # If stuck for too long, do a perturbation
        if no_improve_count > 1000:
            no_improve_count = 0
            # Flip 5-15 random positions
            num_flips = rng.randint(5, 15)
            flip_positions = rng.sample(range(n), num_flips)
            for p in flip_positions:
                current_signs[p] = -current_signs[p]
            current_sums = _compute_ap_sums(current_signs, ap_matrix, ap_lengths)
            current_max = int(np.max(np.abs(current_sums))) if num_aps > 0 else 1
        
        # Early termination if we've reached a very good bound
        if current_max <= 2:
            break
    
    return [int(x) for x in best_signs], best_max

def _construct_signs_sa():
    """
    Construct signs using Thue-Morse init + simulated annealing with multi-start.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_matrix, ap_lengths = _generate_ap_matrix(n)
    num_aps = ap_lengths.shape[0]
    
    if num_aps == 0:
        return [1] * n
    
    # Precompute AP containment for fast lookup
    ap_containing = _precompute_ap_containing_fast(ap_matrix, ap_lengths, n)
    
    # Multi-start: try several different initializations
    best_overall_signs = None
    best_overall_max = float('inf')
    
    # Number of starts based on available time
    time_per_start = 12.0  # seconds per start
    num_starts = max(1, int((deadline - start_time) // time_per_start))
    
    for start_idx in range(num_starts):
        start_time_this = time.time()
        deadline_this = min(deadline, start_time_this + time_per_start - 1)
        
        # Different initialization for each start
        if start_idx == 0:
            # Thue-Morse
            signs = _thue_morse(n)
        elif start_idx == 1:
            # Random
            np_rng = np.random.RandomState(start_idx * 12345 + 67890)
            signs = np_rng.choice([-1, 1], size=n).astype(np.int8)
        else:
            # Perturbed Thue-Morse
            rng_init = random.Random(start_idx * 1000 + 42)
            signs = _thue_morse(n).copy()
            num_flips = rng_init.randint(10, 30)
            flip_positions = rng_init.sample(range(n), num_flips)
            for p in flip_positions:
                signs[p] = -signs[p]
        
        rng = random.Random(start_idx * 12345 + 67890)
        
        result_signs, result_max = _simulate_annealing(
            signs, ap_matrix, ap_lengths, ap_containing, start_time_this, deadline_this, rng
        )
        
        if result_max < best_overall_max:
            best_overall_max = result_max
            best_overall_signs = result_signs
        
        # If we've found a very good solution, stop early
        if best_overall_max <= 2:
            break
    
    if best_overall_signs is None:
        # Fallback to Thue-Morse
        signs = _thue_morse(n)
        return [int(x) for x in signs]
    
    return best_overall_signs

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    signs = _construct_signs_sa()
    return signs

# EVOLVE-BLOCK-END
