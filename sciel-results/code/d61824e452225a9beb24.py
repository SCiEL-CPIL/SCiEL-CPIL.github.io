"""Optimized two-colouring of [1,256] to minimize monochromatic 4-APs.

Structural change: vectorized delta computation using numpy to evaluate
all 256 flip deltas simultaneously per sweep, enabling 300+ sweeps per
initialization. Combined with simulated annealing for escaping local minima.
"""
import time
import random
import numpy as np

N = 256

def _precompute_aps():
    """Return numpy array of shape (M, 4) with all valid 4-APs (0-indexed)."""
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return np.array(aps, dtype=np.int32)

def _count_mono_vec(colors, aps):
    """Count monochromatic 4-APs using numpy. colors is array of 0/1."""
    a, b, c, d = aps[:, 0], aps[:, 1], aps[:, 2], aps[:, 3]
    ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
    mono = (ca == cb) & (cb == cc) & (cc == cd)
    return int(mono.sum())

def _thue_morse():
    """Generate Thue-Morse sequence of length 256 (0-indexed)."""
    tm = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def _compute_all_deltas(colors, aps):
    """
    For each position i in 0..255, compute the change in mono-AP count
    if we flip bit i. Returns array of shape (N,).
    
    Key insight: flipping position i affects only APs containing i.
    For an AP (a,b,c,d):
      - If currently mono (all same color): flipping one element makes it non-mono. delta = -1.
      - If currently non-mono: flipping one element may make it mono. 
        Specifically, if the other 3 elements are all the same color as the NEW value of i,
        then it becomes mono. delta = +1.
    """
    M = len(aps)
    a, b, c, d = aps[:, 0], aps[:, 1], aps[:, 2], aps[:, 3]
    ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
    
    # Current mono status
    is_mono = (ca == cb) & (cb == cc) & (cc == cd)
    
    # For each position, count APs where flipping it would make mono (gain)
    # and APs where it's currently mono (loss)
    deltas = np.zeros(N, dtype=np.int32)
    
    # For position a: flip changes ca. 
    # If is_mono: losing 1 mono AP.
    # If not is_mono: if cb==cc==cd and (1-ca)==cb, then gaining 1.
    # i.e., if the other three are all equal to the new value of ca.
    
    # Process each of the 4 positions in each AP
    # Position 0 (a):
    other_same = (cb == cc) & (cc == cd)
    # After flip, ca becomes 1-ca. It becomes mono if other_same AND (1-ca)==cb
    would_be_mono_if_flip_a = other_same & (cb == (1 - ca))
    deltas[a] += is_mono.astype(np.int32) * (-1) + would_be_mono_if_flip_a.astype(np.int32)
    
    # Position 1 (b):
    other_same_b = (ca == cc) & (cc == cd)
    would_be_mono_if_flip_b = other_same_b & (ca == (1 - cb))
    deltas[b] += is_mono.astype(np.int32) * (-1) + would_be_mono_if_flip_b.astype(np.int32)
    
    # Position 2 (c):
    other_same_c = (ca == cb) & (cb == cd)
    would_be_mono_if_flip_c = other_same_c & (ca == (1 - cc))
    deltas[c] += is_mono.astype(np.int32) * (-1) + would_be_mono_if_flip_c.astype(np.int32)
    
    # Position 3 (d):
    other_same_d = (ca == cb) & (cb == cc)
    would_be_mono_if_flip_d = other_same_d & (ca == (1 - cd))
    deltas[d] += is_mono.astype(np.int32) * (-1) + would_be_mono_if_flip_d.astype(np.int32)
    
    return deltas

def _vectorized_local_search(colors, aps, max_sweeps=500, deadline=None, use_annealing=False, T_init=5.0, T_final=0.1):
    """
    Vectorized local search with optional simulated annealing.
    Returns (best_coloring_array, best_count).
    """
    current = colors.copy()
    current_count = _count_mono_vec(current, aps)
    best = current.copy()
    best_count = current_count
    
    rng = np.random.default_rng(42)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all deltas at once
        deltas = _compute_all_deltas(current, aps)
        
        if use_annealing:
            # Simulated annealing: accept moves with probability exp(-delta/T)
            T = T_init * (T_final / T_init) ** (sweep / max(1, max_sweeps))
            # For each position, decide whether to flip
            # We want to flip if delta < 0 (always good) or with prob exp(-delta/T) if delta > 0
            # To keep it simple and fast, do greedy + random perturbation
            # Find best improving move
            min_delta = deltas.min()
            if min_delta < 0:
                # Flip the position with most negative delta (best improvement)
                pos = int(deltas.argmin())
                current[pos] = 1 - current[pos]
                current_count += int(min_delta)
            else:
                # No improving move; accept a random worsening move with prob
                # Pick a random position
                pos = int(rng.integers(0, N))
                delta_pos = int(deltas[pos])
                accept_prob = np.exp(-delta_pos / T) if delta_pos > 0 else 1.0
                if rng.random() < accept_prob:
                    current[pos] = 1 - current[pos]
                    current_count += delta_pos
        else:
            # Greedy: flip the best improving position if any
            min_delta = int(deltas.min())
            if min_delta < 0:
                pos = int(deltas.argmin())
                current[pos] = 1 - current[pos]
                current_count += min_delta
            else:
                # No improving move; try random perturbation to escape
                # Flip 2-3 random positions
                n_perturb = int(rng.integers(2, 5))
                for _ in range(n_perturb):
                    pos = int(rng.integers(0, N))
                    current[pos] = 1 - current[pos]
                current_count = _count_mono_vec(current, aps)
        
        if current_count < best_count:
            best_count = current_count
            best = current.copy()
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    aps = _precompute_aps()
    
    # Multiple initializations
    initializations = []
    
    # Thue-Morse
    initializations.append(_thue_morse())
    
    # Random colorings
    rng = random.Random(42)
    for _ in range(10):
        initializations.append(np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8))
    
    # Structured patterns
    for block in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        initializations.append(np.array([(i // block) % 2 for i in range(N)], dtype=np.int8))
    
    # XOR-based patterns
    for shift in [1, 2, 3, 4, 5, 6, 7, 8]:
        pattern = np.array([(i >> shift) % 2 for i in range(N)], dtype=np.int8)
        initializations.append(pattern)
    
    # Thue-Morse variants (shifted)
    tm = _thue_morse()
    for shift in [1, 2, 3, 4, 5, 6, 7, 8]:
        initializations.append(np.roll(tm, shift))
    
    best_coloring = None
    best_count = float('inf')
    
    # Process each initialization with vectorized local search
    for init_idx, init_colors in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Quick count first
        init_count = _count_mono_vec(init_colors, aps)
        if init_count < best_count:
            best_count = init_count
            best_coloring = init_colors.copy()
        
        # Run vectorized local search
        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break
        
        # Allocate time per initialization
        time_per_init = min(5.0, remaining_time / max(1, len(initializations) - init_idx))
        init_deadline = time.time() + time_per_init
        
        # First pass: pure greedy
        lc, lc_count = _vectorized_local_search(
            init_colors, aps, max_sweeps=300, 
            deadline=init_deadline, use_annealing=False
        )
        if lc_count < best_count:
            best_count = lc_count
            best_coloring = lc.copy()
        
        # Second pass: with annealing if time remains
        if time.time() < deadline - 2:
            lc2, lc2_count = _vectorized_local_search(
                lc, aps, max_sweeps=200,
                deadline=min(init_deadline + 3.0, deadline - 2),
                use_annealing=True, T_init=3.0, T_final=0.05
            )
            if lc2_count < best_count:
                best_count = lc2_count
                best_coloring = lc2.copy()
    
    # Final polish: run more sweeps on the best solution
    if best_coloring is not None and time.time() < deadline - 5:
        final_deadline = deadline - 2
        lc_final, lc_final_count = _vectorized_local_search(
            best_coloring, aps, max_sweeps=1000,
            deadline=final_deadline, use_annealing=True, T_init=2.0, T_final=0.01
        )
        if lc_final_count < best_count:
            best_count = lc_final_count
            best_coloring = lc_final
    
    # Return as list of ints (1-indexed mapping: position i in 0..255 corresponds to integer i+1)
    return [int(c) for c in best_coloring]
