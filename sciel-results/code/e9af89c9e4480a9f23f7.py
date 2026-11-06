"""Two-phase strategy: fast critical-AP search + full-constraint polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
DEADLINE_MARGIN = 3.0  # seconds of safety margin before the 90s kill

def _build_incidence():
    """Build the binary incidence matrix for all APs in [1..256].
    Rows = APs, columns = positions 0..255.
    Entry (r, c) = 1 if position c+1 is in AP r.
    """
    rows = []
    for start in range(1, N + 1):
        max_step = N - start + 1
        for step in range(1, max_step + 1):
            length = (N - start) // step + 1
            if length < 1:
                continue
            row = np.zeros(N, dtype=np.float64)
            for k in range(length):
                pos = start + k * step - 1
                if pos < N:
                    row[pos] = 1.0
            rows.append(row)
    return np.array(rows, dtype=np.float64)

# Precompute the full incidence matrix at module load time
_INC_FULL = _build_incidence()
_NUM_APS_FULL = _INC_FULL.shape[0]

# Precompute list of AP indices for each position for faster updates (full)
_APS_PER_POS_FULL = [np.nonzero(_INC_FULL[:, j] == 1.0)[0] for j in range(N)]

def _identify_critical_aps(signs, top_k=200):
    """Identify the top_k APs with largest absolute sums given current signs.
    These are the 'critical' constraints that dominate the max.
    """
    sums = _INC_FULL @ signs
    abs_sums = np.abs(sums)
    # Get indices of top_k largest absolute sums
    if top_k >= len(abs_sums):
        critical_idx = np.arange(len(abs_sums))
    else:
        # Use argpartition for efficiency
        part_idx = np.argpartition(abs_sums, -top_k)[-top_k:]
        critical_idx = part_idx
    return critical_idx

def _build_critical_incidence(critical_idx):
    """Build incidence matrix using only critical APs."""
    return _INC_FULL[critical_idx, :]

def _max_abs_sum_full(signs):
    """Compute max absolute signed sum over all APs (full constraint set)."""
    sums = _INC_FULL @ signs
    return float(np.max(np.abs(sums)))

def construct():
    t_start = time.time()
    deadline = 90.0 - DEADLINE_MARGIN

    # Default fallback: alternating signs
    default_signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    
    best_signs = default_signs.copy()
    best_score = _max_abs_sum_full(best_signs)

    rng = np.random.default_rng(42)

    # Phase 1: Fast search using critical APs
    # Phase 2: Polish with full constraints
    
    # Phase 1: Use critical APs for fast exploration
    # Start with random signs, identify critical APs, then do SA on those
    max_phase1_restarts = 30
    max_phase2_polish_steps = 5000
    
    for restart in range(max_phase1_restarts):
        if time.time() > deadline:
            break
        
        # Initialize signs
        if restart == 0:
            signs = rng.choice([-1.0, 1.0], size=N)
        elif restart % 3 == 0:
            signs = rng.choice([-1.0, 1.0], size=N)
        else:
            # Perturb best found so far
            signs = best_signs.copy()
            num_flips = rng.integers(3, 15)
            flip_idx = rng.choice(N, size=num_flips, replace=False)
            signs[flip_idx] *= -1.0
        
        # Identify critical APs for this starting point
        critical_idx = _identify_critical_aps(signs, top_k=300)
        inc_crit = _build_critical_incidence(critical_idx)
        aps_per_pos_crit = [np.nonzero(inc_crit[:, j] == 1.0)[0] for j in range(N)]
        
        # SA on critical APs
        sums_crit = inc_crit @ signs
        current_score = float(np.max(np.abs(sums_crit)))
        
        T = 3.0
        cooling_rate = 0.9995
        max_steps = 20000
        step_count = 0
        
        best_local_signs = signs.copy()
        best_local_score = current_score
        
        while True:
            if time.time() > deadline:
                break
            if step_count > max_steps:
                break
            
            step_count += 1
            
            j = rng.integers(0, N)
            old_val = signs[j]
            
            idxs = aps_per_pos_crit[j]
            if len(idxs) > 0:
                sums_crit[idxs] -= 2.0 * old_val
            
            signs[j] = -old_val
            
            new_score = float(np.max(np.abs(sums_crit)))
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_signs = signs.copy()
            else:
                signs[j] = old_val
                if len(idxs) > 0:
                    sums_crit[idxs] += 2.0 * old_val
            
            T *= cooling_rate
            if T < 0.01:
                break
        
        # Evaluate on FULL constraint set
        full_score = _max_abs_sum_full(best_local_signs)
        if full_score < best_score:
            best_score = full_score
            best_signs = best_local_signs.copy()
    
    # Phase 2: Polish with full constraints
    if time.time() < deadline:
        signs = best_signs.copy()
        sums_full = _INC_FULL @ signs
        current_score = float(np.max(np.abs(sums_full)))
        
        T = 1.0
        cooling_rate = 0.999
        max_steps = max_phase2_polish_steps
        step_count = 0
        
        best_local_signs = signs.copy()
        best_local_score = current_score
        
        while True:
            if time.time() > deadline:
                break
            if step_count > max_steps:
                break
            
            step_count += 1
            
            j = rng.integers(0, N)
            old_val = signs[j]
            
            idxs = _APS_PER_POS_FULL[j]
            if len(idxs) > 0:
                sums_full[idxs] -= 2.0 * old_val
            
            signs[j] = -old_val
            
            new_score = float(np.max(np.abs(sums_full)))
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_signs = signs.copy()
            else:
                signs[j] = old_val
                if len(idxs) > 0:
                    sums_full[idxs] += 2.0 * old_val
            
            T *= cooling_rate
            if T < 0.005:
                break
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_signs = best_local_signs.copy()

    result = [int(s) for s in best_signs]
    return result

# EVOLVE-BLOCK-END
