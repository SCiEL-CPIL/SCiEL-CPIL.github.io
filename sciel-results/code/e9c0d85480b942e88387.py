"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions with length >= 2
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(N - d):
            if start + d < N:
                indices = list(range(start, N, d))
                if len(indices) >= 2:
                    aps.append(indices)
    
    num_aps = len(aps)
    
    # Build the AP matrix as a numpy array of shape (num_aps, N) with float32
    # Use Fortran order for contiguous columns to speed up column access
    M = np.zeros((num_aps, N), dtype=np.float32, order='F')
    for i, ap in enumerate(aps):
        for idx in ap:
            M[i, idx] = 1.0
    
    Mf = M
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute sum over all APs using matrix-vector product."""
        sums = Mf.dot(signs.astype(np.float32))
        return int(np.max(np.abs(sums)))
    
    rng = random.Random(42)
    
    # Phase 1: Generate diverse initial solutions using random restarts with local search
    best_signs = None
    best_cost = float('inf')
    
    # Try multiple random initializations with quick local search
    for init_seed in range(20):
        if time.time() >= deadline - 20.0:
            break
        
        rng_init = random.Random(init_seed)
        signs = np.array([rng_init.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        
        # Quick single-flip local search to improve this random start
        sums = Mf.dot(signs.astype(np.float32))
        current_cost = int(np.max(np.abs(sums)))
        
        # A few passes of single-flip local search
        for _ in range(3):
            if time.time() >= deadline - 20.0:
                break
            improved = False
            for idx in range(N):
                old_sign = signs[idx]
                new_sign = -old_sign
                signs[idx] = new_sign
                delta = -2.0 * old_sign
                sums += delta * Mf[:, idx]
                new_cost = int(np.max(np.abs(sums)))
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    signs[idx] = old_sign
                    sums -= delta * Mf[:, idx]
            
            if not improved:
                break
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_signs = signs.copy()
    
    # Phase 2: Multi-flip simulated annealing with restarts
    if best_signs is None:
        best_signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        best_cost = compute_max_abs_sum(best_signs)
    
    signs = best_signs.copy()
    current_cost = best_cost
    sums = Mf.dot(signs.astype(np.float32))
    
    # Multi-flip SA: flip 2-4 bits simultaneously to escape local minima
    T = 3.0
    cooling_rate = 0.9998
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10.0:
        if T < 0.05 or no_improve > max_no_improve:
            # Restart with perturbation of best solution
            signs = best_signs.copy()
            # Perturb 5-15 random bits
            n_perturb = rng.randint(5, 15)
            for _ in range(n_perturb):
                idx = rng.randrange(N)
                signs[idx] = -signs[idx]
            sums = Mf.dot(signs.astype(np.float32))
            current_cost = int(np.max(np.abs(sums)))
            T = 3.0
            no_improve = 0
            continue
        
        # Pick 2-4 random indices to flip simultaneously
        n_flips = rng.choice([2, 3, 4])
        flip_indices = rng.sample(range(N), n_flips)
        
        # Apply all flips and compute new cost
        old_sums = sums.copy()
        for idx in flip_indices:
            signs[idx] = -signs[idx]
        
        sums = Mf.dot(signs.astype(np.float32))
        new_cost = int(np.max(np.abs(sums)))
        
        delta_cost = new_cost - current_cost
        
        if new_cost <= current_cost:
            current_cost = new_cost
            no_improve = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-delta_cost / T)
            prob = np.exp(-delta_cost / T)
            if rng.random() < prob:
                current_cost = new_cost
                no_improve += 1
            else:
                # Revert
                signs = signs.copy()
                for idx in flip_indices:
                    signs[idx] = -signs[idx]
                sums = old_sums
                no_improve += 1
        
        T *= cooling_rate
    
    # Final intensive single-flip local search on best solution
    signs = best_signs.copy()
    sums = Mf.dot(signs.astype(np.float32))
    current_cost = best_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            old_sign = signs[idx]
            new_sign = -old_sign
            
            signs[idx] = new_sign
            delta = -2.0 * old_sign
            sums += delta * Mf[:, idx]
            
            new_cost = int(np.max(np.abs(sums)))
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
                current_cost = new_cost
                improved = True
            else:
                signs[idx] = old_sign
                sums -= delta * Mf[:, idx]
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
