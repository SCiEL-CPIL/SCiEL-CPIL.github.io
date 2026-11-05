import numpy as np
import time
import random

def construct():
    N = 256
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute all 4-APs as index arrays
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps = np.array(ap_list, dtype=np.int32)  # shape (M, 4)
    M = len(aps)
    
    # Build incidence: for each position i, which APs does it participate in, and which are the other 3?
    # We'll use a different approach: build a matrix A of shape (N, M) where A[i, m] = 1 if position i is in AP m
    # But for delta computation, we need: for position i, sum over APs containing i of (product of other 3 colors)
    # If cols[i] == 0 (using 0/1): mono if all 4 are 0 OR all 4 are 1
    # Let's use signed representation: x[i] = +1 for color 0, -1 for color 1
    # Then a 4-AP is monochromatic iff x[i1]*x[i2]*x[i3]*x[i4] = +1
    # Objective: minimize count of APs with product +1
    # Equivalently: minimize sum over APs of (1 + prod)/2 = M/2 + (1/2)*sum(prod)
    # So minimizing mono count <=> minimizing sum of products over all APs
    
    # Build the product structure. For each AP m with indices (a, b, c, d):
    # prod = x[a]*x[b]*x[c]*x[d]
    # Delta of flipping x[i]: if i is in AP m, new_prod = -old_prod
    # So delta for flipping i = sum over APs containing i of (new_prod - old_prod) = -2 * sum over APs containing i of old_prod
    # Wait: old_prod = x[i] * (product of others). new_prod = -x[i] * (product of others) = -old_prod.
    # delta = new_prod - old_prod = -2 * old_prod = -2 * x[i] * (prod of others)
    
    # For vectorized computation:
    # For each position i, we need: S[i] = sum over APs m containing i of (product of the other 3 x-values)
    # Then delta[i] = -2 * x[i] * S[i]
    
    # Build: for each AP m, it contributes to S at its 4 positions.
    # S[a] += x[b]*x[c]*x[d]
    # S[b] += x[a]*x[c]*x[d]
    # S[c] += x[a]*x[b]*x[d]
    # S[d] += x[a]*x[b]*x[c]
    
    # We can do this with numpy: for each AP, compute the 4 "other product" terms.
    # Let X = x[aps]  # shape (M, 4)
    # prod_all = X.prod(axis=1)  # shape (M,)
    # other_prod for position j in AP m = prod_all[m] / X[m, j]  (but x is ±1, so division = multiplication)
    # other_prod[m, j] = prod_all[m] * X[m, j]  (since 1/x = x for ±1)
    
    # So: other_prods = prod_all[:, None] * X  # shape (M, 4)
    # Then S[i] = sum over m where i is at position j of other_prods[m, j]
    
    # Let's build this efficiently.
    # positions_in_aps: aps has shape (M, 4). 
    # For column j (j=0,1,2,3), the positions are aps[:, j]
    # S[aps[:, j]] += other_prods[:, j]
    
    def compute_S(x):
        """Compute S[i] = sum over APs containing i of product of other 3 x-values."""
        X = x[aps]  # (M, 4)
        prod_all = X.prod(axis=1)  # (M,)
        other_prods = prod_all[:, None] * X  # (M, 4) since 1/x = x
        S = np.zeros(N, dtype=np.int32)
        for j in range(4):
            np.add.at(S, aps[:, j], other_prods[:, j])
        return S
    
    def count_mono_from_x(x):
        """Count monochromatic 4-APs from signed representation x (±1)."""
        X = x[aps]
        prod_all = X.prod(axis=1)
        # mono iff prod == +1
        return int(np.sum(prod_all == 1))
    
    def count_mono_from_bits(bits):
        """Count monochromatic 4-APs from 0/1 bits."""
        x = 1 - 2 * bits  # 0 -> +1, 1 -> -1
        return count_mono_from_x(x)
    
    def bits_to_x(bits):
        return 1 - 2 * bits.astype(np.int8)
    
    def x_to_bits(x):
        return ((1 - x) // 2).astype(np.int8)
    
    def vectorized_greedy_step(bits, max_rounds=50):
        """Apply greedy flips: compute all deltas, flip all with negative delta, repeat."""
        x = bits_to_x(bits)
        for _ in range(max_rounds):
            S = compute_S(x)
            deltas = -2 * x * S  # delta in objective (sum of products)
            # In terms of mono count: delta_mono = -delta/2 ... let's just track sum of products
            # Actually, let's track the raw sum of products as the objective
            flip_mask = deltas < 0
            if not np.any(flip_mask):
                break
            x[flip_mask] *= -1
        return x_to_bits(x)
    
    def batched_sa(bits, T_start, cooling, min_T, time_limit):
        """Simulated annealing with batched flips using vectorized delta computation."""
        x = bits_to_x(bits)
        current_obj = int(np.sum((x[aps]).prod(axis=1)))  # sum of products
        best_obj = current_obj
        best_x = x.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            S = compute_S(x)
            deltas = -2 * x * S  # delta in sum-of-products objective
            
            # Batched stochastic step: for each position, decide to flip with probability
            # based on delta and temperature
            # delta < 0: always flip (improvement)
            # delta > 0: flip with prob exp(-delta/T)
            # delta == 0: flip with prob 0.5 (random walk)
            
            flip_mask = np.zeros(N, dtype=bool)
            neg_mask = deltas < 0
            flip_mask[neg_mask] = True
            
            pos_mask = deltas > 0
            if np.any(pos_mask):
                if T > min_T:
                    probs = np.exp(-deltas[pos_mask].astype(np.float64) / T)
                    rand_vals = np.random.random(np.sum(pos_mask))
                    flip_indices = np.where(pos_mask)[0]
                    flip_mask[flip_indices[rand_vals < probs]] = True
                # else: don't flip positive deltas at low T
            
            zero_mask = (deltas == 0)
            if np.any(zero_mask):
                rand_vals = np.random.random(np.sum(zero_mask))
                zero_indices = np.where(zero_mask)[0]
                flip_mask[zero_indices[rand_vals < 0.5]] = True
            
            if np.any(flip_mask):
                x[flip_mask] *= -1
                # Recompute objective
                current_obj = int(np.sum((x[aps]).prod(axis=1)))
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return x_to_bits(best_x), best_obj
    
    def perturb(bits, num_flips):
        """Randomly flip num_flips positions."""
        new_bits = bits.copy()
        indices = np.random.choice(N, size=num_flips, replace=False)
        new_bits[indices] = 1 - new_bits[indices]
        return new_bits
    
    # Initial colorings
    global_best_bits = None
    global_best_obj = float('inf')
    
    initializations = []
    # All zeros
    initializations.append(np.zeros(N, dtype=np.int8))
    # All ones
    initializations.append(np.ones(N, dtype=np.int8))
    # Alternating
    initializations.append(np.array([i % 2 for i in range(N)], dtype=np.int8))
    # Block pattern (period 4)
    initializations.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))
    # Block pattern (period 8)
    initializations.append(np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8))
    # Random
    for _ in range(3):
        initializations.append(np.random.randint(0, 2, size=N, dtype=np.int8))
    
    time_per_init = 130.0 / len(initializations)
    
    for init_idx, init_bits in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # First: greedy polish from this initialization
        greedy_bits = vectorized_greedy_step(init_bits, max_rounds=100)
        greedy_obj = count_mono_from_bits(greedy_bits)
        
        if greedy_obj < global_best_obj:
            global_best_obj = greedy_obj
            global_best_bits = greedy_bits.copy()
        
        # Then: batched SA starting from the greedy result
        T_start = 2.0 + 0.5 * (init_idx % 4)
        cooling = 0.99995
        min_T = 0.01
        
        sa_bits, sa_obj = batched_sa(
            greedy_bits, T_start, cooling, min_T, time_per_init * 0.7
        )
        
        # Final greedy polish on SA result
        polished_bits = vectorized_greedy_step(sa_bits, max_rounds=100)
        polished_obj = count_mono_from_bits(polished_bits)
        
        if polished_obj < global_best_obj:
            global_best_obj = polished_obj
            global_best_bits = polished_bits.copy()
        
        # Also try perturbing the global best and polishing
        if init_idx >= 2:
            pert_bits = perturb(global_best_bits, random.randint(8, 32))
            pert_greedy_bits = vectorized_greedy_step(pert_bits, max_rounds=50)
            pert_greedy_obj = count_mono_from_bits(pert_greedy_bits)
            if pert_greedy_obj < global_best_obj:
                global_best_obj = pert_greedy_obj
                global_best_bits = pert_greedy_bits.copy()
    
    # Final polish
    if global_best_bits is not None:
        final_bits = vectorized_greedy_step(global_best_bits, max_rounds=200)
        final_obj = count_mono_from_bits(final_bits)
        if final_obj < global_best_obj:
            global_best_obj = final_obj
            global_best_bits = final_bits
    
    # Convert to list of 0/1 ints
    return [int(b) for b in global_best_bits]

# Pre-warm: compute aps at module level to avoid recomputation
# Actually, let's keep it inside construct to be safe, but we can cache
_aps_cache = None

def _get_aps():
    global _aps_cache
    if _aps_cache is None:
        N = 256
        ap_list = []
        for d in range(1, N // 3 + 1):
            max_a = N - 3 * d
            for a in range(max_a):
                ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
        _aps_cache = np.array(ap_list, dtype=np.int32)
    return _aps_cache

# Override to use cached APs
def construct():
    N = 256
    deadline = time.time() + 140
    
    aps = _get_aps()
    M = len(aps)
    
    def compute_S(x):
        X = x[aps]
        prod_all = X.prod(axis=1)
        other_prods = prod_all[:, None] * X
        S = np.zeros(N, dtype=np.int32)
        for j in range(4):
            np.add.at(S, aps[:, j], other_prods[:, j])
        return S
    
    def count_mono_from_bits(bits):
        x = 1 - 2 * bits.astype(np.int8)
        X = x[aps]
        prod_all = X.prod(axis=1)
        return int(np.sum(prod_all == 1))
    
    def bits_to_x(bits):
        return (1 - 2 * bits.astype(np.int8)).astype(np.int8)
    
    def x_to_bits(x):
        return ((1 - x) // 2).astype(np.int8)
    
    def vectorized_greedy_step(bits, max_rounds=50):
        x = bits_to_x(bits)
        for _ in range(max_rounds):
            S = compute_S(x)
            deltas = -2 * x * S
            flip_mask = deltas < 0
            if not np.any(flip_mask):
                break
            x[flip_mask] *= -1
        return x_to_bits(x)
    
    def batched_sa(bits, T_start, cooling, min_T, time_limit):
        x = bits_to_x(bits)
        X = x[aps]
        current_obj = int(np.sum(X.prod(axis=1)))
        best_obj = current_obj
        best_x = x.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            S = compute_S(x)
            deltas = -2 * x * S
            
            flip_mask = np.zeros(N, dtype=bool)
            neg_mask = deltas < 0
            flip_mask[neg_mask] = True
            
            pos_mask = deltas > 0
            if np.any(pos_mask):
                if T > min_T:
                    probs = np.exp(-deltas[pos_mask].astype(np.float64) / T)
                    rand_vals = np.random.random(np.sum(pos_mask))
                    flip_indices = np.where(pos_mask)[0]
                    flip_mask[flip_indices[rand_vals < probs]] = True
            
            zero_mask = (deltas == 0)
            if np.any(zero_mask):
                rand_vals = np.random.random(np.sum(zero_mask))
                zero_indices = np.where(zero_mask)[0]
                flip_mask[zero_indices[rand_vals < 0.5]] = True
            
            if np.any(flip_mask):
                x[flip_mask] *= -1
                X = x[aps]
                current_obj = int(np.sum(X.prod(axis=1)))
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return x_to_bits(best_x), best_obj
    
    def perturb(bits, num_flips):
        new_bits = bits.copy()
        indices = np.random.choice(N, size=num_flips, replace=False)
        new_bits[indices] = 1 - new_bits[indices]
        return new_bits
    
    global_best_bits = None
    global_best_obj = float('inf')
    
    initializations = [
        np.zeros(N, dtype=np.int8),
        np.ones(N, dtype=np.int8),
        np.array([i % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8),
        np.random.randint(0, 2, size=N, dtype=np.int8),
        np.random.randint(0, 2, size=N, dtype=np.int8),
        np.random.randint(0, 2, size=N, dtype=np.int8),
    ]
    
    time_per_init = 130.0 / len(initializations)
    
    for init_idx, init_bits in enumerate(initializations):
        if time.time() > deadline:
            break
        
        greedy_bits = vectorized_greedy_step(init_bits, max_rounds=100)
        greedy_obj = count_mono_from_bits(greedy_bits)
        
        if greedy_obj < global_best_obj:
            global_best_obj = greedy_obj
            global_best_bits = greedy_bits.copy()
        
        T_start = 2.0 + 0.5 * (init_idx % 4)
        cooling = 0.99995
        min_T = 0.01
        
        sa_bits, sa_obj = batched_sa(
            greedy_bits, T_start, cooling, min_T, time_per_init * 0.7
        )
        
        polished_bits = vectorized_greedy_step(sa_bits, max_rounds=100)
        polished_obj = count_mono_from_bits(polished_bits)
        
        if polished_obj < global_best_obj:
            global_best_obj = polished_obj
            global_best_bits = polished_bits.copy()
        
        if init_idx >= 2:
            pert_bits = perturb(global_best_bits, random.randint(8, 32))
            pert_greedy_bits = vectorized_greedy_step(pert_bits, max_rounds=50)
            pert_greedy_obj = count_mono_from_bits(pert_greedy_bits)
            if pert_greedy_obj < global_best_obj:
                global_best_obj = pert_greedy_obj
                global_best_bits = pert_greedy_bits.copy()
    
    if global_best_bits is not None:
        final_bits = vectorized_greedy_step(global_best_bits, max_rounds=200)
        final_obj = count_mono_from_bits(final_bits)
        if final_obj < global_best_obj:
            global_best_obj = final_obj
            global_best_bits = final_bits
    
    return [int(b) for b in global_best_bits]
