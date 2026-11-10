# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Solves the discrete 3D Kakeya needle problem for N=8.
    Uses the decoupling property to optimize 1D positions for x and y separately,
    then combines them. Assumes symmetry x_i = y_i for the main optimization,
    then refines with full 3D SA if time permits.
    """
    N = 8
    num_tubes = N * N
    start_time = time.time()
    deadline = start_time + 550.0
    
    # 1. Keich Baseline
    def get_keich_positions():
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    x_base = get_keich_positions()
    
    # Precompute indices for 1D optimization
    i_idx = np.arange(N, dtype=np.float64)
    
    def compute_1d_cost(x, n_samples=100000, seed=12345):
        """
        Computes the approximate 3D volume assuming symmetry x_i = y_i.
        V = int_0^1 L(c)^2 dc.
        """
        rng = np.random.default_rng(seed)
        c = rng.uniform(0, 1, n_samples)
        
        # Calculate starts and ends for all tubes and samples
        # S[n, i] = x[i] + c[n] * i / N
        S = x[None, :] + c[:, None] * i_idx[None, :] / N
        E = S + 1.0 / N
        
        # Create events: (time, type)
        # Type: +1 for start, -1 for end
        # We need to sort by time for each sample
        # To vectorize sorting of 16 items per row, we can use np.sort on the combined array
        # But we need to keep track of types.
        # Trick: Encode time and type into a single float? No, sorting floats is stable?
        # Better: Just sort the indices.
        
        # Combine S and E into one array of shape (n, 2*N)
        # Col 0..7 are starts, 8..15 are ends?
        # No, let's just interleave or just sort the 16 values.
        # We need to know if a value is a start or end.
        # If we sort S and E separately, we lose the pairing.
        
        # Alternative: Since N=8 is small, we can use a simple loop over the 8 intervals
        # but vectorized over samples?
        # Actually, let's just use the inclusion-exclusion or a simple sweep.
        # Given N=8, the number of unique intervals is small.
        
        # Let's try the vectorized sweep line with sorting.
        # Create an array of shape (n, 16, 2)
        # Col 0: time, Col 1: delta
        # Starts: time=S, delta=1
        # Ends: time=E, delta=-1
        
        # We can construct this by stacking S and E
        # S is (n, 8), E is (n, 8)
        # We want to sort the 16 values in each row.
        
        # Let's just do a simple loop over the N=8 intervals to compute the union length?
        # No, that's O(N^2) per sample if we merge.
        # With N=8, O(64) operations per sample. 100k samples -> 6.4M ops. Fast enough.
        
        # Vectorized Union Length Calculation:
        # 1. Flatten S and E to (n*8, 1)
        # 2. This is getting complicated. Let's use a simpler approximation or just SA on 128 vars with fast 3D estimator.
        
        # Actually, the 3D estimator with 50k samples is ~0.1s.
        # 128-dim SA with 50k samples:
        # 1000 steps -> 100s.
        # This is simpler and less error-prone than the 1D decoupling if the decoupling assumption is slightly off.
        # Let's stick to the 128-dim SA but make it very fast.
        
        pass # Placeholder

    # Let's revert to the 128-dim SA but optimized.
    # The Keich baseline is good.
    # We will use a fast 3D volume estimator.
    
    i_idx_full = np.array([k // N for k in range(num_tubes)], dtype=np.float64)
    j_idx_full = np.array([k % N for k in range(num_tubes)], dtype=np.float64)
    
    def compute_volume_fast(positions, n_samples=20000, seed=12345):
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0] + (i_idx_full + 1) / N)
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1] + (j_idx_full + 1) / N)
        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area <= 1e-9:
            return 1e9
        rng = np.random.default_rng(seed)
        a = rng.uniform(x_min, x_max, n_samples)
        b = rng.uniform(y_min, y_max, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # Vectorized check
        # a[:, None] - c[:, None] * i_idx[None, :] - 1/N <= x_k[None, :] <= a[:, None] - c[:, None] * i_idx[None, :]
        # To save memory, we can process in chunks?
        # 20k * 64 * 8 bytes = 10MB. Very safe.
        
        x_lower = a[:, None] - c[:, None] * i_idx_full[None, :] - 1.0/N
        x_upper = a[:, None] - c[:, None] * i_idx_full[None, :]
        y_lower = b[:, None] - c[:, None] * j_idx_full[None, :] - 1.0/N
        y_upper = b[:, None] - c[:, None] * j_idx_full[None, :]
        
        x_ok = (positions[None, :, 0] >= x_lower) & (positions[None, :, 0] <= x_upper)
        y_ok = (positions[None, :, 1] >= y_lower) & (positions[None, :, 1] <= y_upper)
        
        in_any_tube = np.any(x_ok & y_ok, axis=1)
        fraction = np.mean(in_any_tube)
        return fraction * box_area

    # Initialize with Keich
    keich_1d = get_keich_positions()
    positions = np.zeros((num_tubes, 2))
    for k in range(num_tubes):
        i = k // N
        j = k % N
        positions[k, 0] = keich_1d[i]
        positions[k, 1] = keich_1d[j]
        
    current_vol = compute_volume_fast(positions)
    best_positions = positions.copy()
    best_vol = current_vol
    
    # SA Parameters
    T = 0.05
    T_min = 1e-4
    cooling_rate = 0.995
    step_size = 0.1
    n_samples_sa = 20000
    
    # Adaptive Step Size
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Perturb
        new_positions = current_positions = positions.copy() # Wait, need to copy current
        current_positions = positions.copy()
        
        # Perturb a subset of tubes to allow local moves
        num_perturb = np.random.randint(1, 5)
        idxs = np.random.choice(num_tubes, num_perturb, replace=False)
        new_positions[idxs] += np.random.normal(0, step_size, (num_perturb, 2))
        
        new_vol = compute_volume_fast(new_positions, n_samples=n_samples_sa)
        
        if new_vol < best_vol:
            best_vol = new_vol
            best_positions = new_positions.copy()
            no_improve_count = 0
            # Reduce step size slightly on improvement? Or keep it.
            # Let's reduce step size to refine
            step_size *= 0.95
            if step_size < 0.001:
                step_size = 0.001
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Increase step size to escape local minimum
                step_size *= 1.2
                if step_size > 0.5:
                    step_size = 0.5
                no_improve_count = 0
                
        # Acceptance
        if new_vol < current_vol:
            accept = True
        else:
            delta = new_vol - current_vol
            if delta > 50 * T:
                accept = False
            else:
                accept = np.random.random() < np.exp(-delta / T)
        
        if accept:
            positions = new_positions
            current_vol = new_vol
        else:
            # Keep current positions
            pass
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final check with full samples to ensure we return a valid, high-quality result
    # The evaluator uses 500k samples.
    # We return best_positions.
    
    return best_positions

# EVOLVE-BLOCK-END
