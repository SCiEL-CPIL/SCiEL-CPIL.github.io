import time
import random
import numpy as np
from scipy.fft import fft, ifft, fftfreq

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_SIZE = TARGET_MAX - TARGET_MIN + 1
    
    # Smooth cost function using log-sum-exp for peak approximation
    def get_cost(sums, smooth_beta=50.0):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1].astype(float)
        
        # Penalty for gaps
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 10000 + gaps * 100 + np.max(target_sums)
            
        # Smooth peak using log-sum-exp: approx max via log(sum(exp(beta*x)))
        # This is differentiable and smoother than hard max
        scaled = target_sums * smooth_beta
        max_val = np.max(scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        smooth_peak = log_sum_exp / smooth_beta
        
        # Secondary: sum of squares to flatten
        sq = np.sum(target_sums**2)
        
        return smooth_peak * 1000 + sq

    # Hard cost for final evaluation
    def get_hard_cost(sums):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1]
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 10000 + gaps * 100 + np.max(target_sums)
        peak = np.max(target_sums)
        sq = np.sum(target_sums**2)
        return peak * 1000 + sq

    # Helper to update sums for adding/removing x
    def apply_add(A, sums, x):
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        if 2*x <= 256:
            sums[2*x] += 1
        A.add(x)
        
    def apply_remove(A, sums, x):
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        if 2*x <= 256:
            sums[2*x] -= 1
        A.remove(x)

    # Phase retrieval via Gerchberg-Saxton in frequency domain
    # Binary vector x of length 257 (indices 0..256), support [0,128]
    # Autocorrelation R[k] = sum_i x[i]*x[i+k] for k>=0
    # We want R[k] ~ 1 for k in [128,256] (each target sum has ~1 representation)
    def gerchberg_saxton(x_init, n_iters=80, target_level=1.0, support_max=128):
        """
        Phase retrieval: given magnitude of autocorrelation in target range,
        recover a binary (or near-binary) vector x with support [0, support_max].
        """
        L = 257  # length of x
        x = x_init.astype(float).copy()
        # Ensure support
        x[support_max+1:] = 0.0
        
        for it in range(n_iters):
            # Compute autocorrelation via FFT
            X = fft(x, n=2*L)
            R = ifft(X * np.conj(X), n=2*L).real[:L]
            # R[k] = sum_i x[i]*x[i+k] for k=0..L-1
            
            # In frequency domain: |X|^2 is fixed magnitude
            # Update: keep |X|, adjust phase to match target autocorrelation
            target_R = np.zeros(L)
            target_R[128:257] = target_level
            target_R[0] = np.sum(x**2)  # keep energy
            
            # Gerchberg-Saxton step:
            # 1. Compute current R from x
            # 2. Blend R toward target_R in the target range
            # 3. Compute X from modified R, keep phase of original X, update x
            # Actually standard GS: 
            # - In time domain: enforce support (zero outside [0,128])
            # - In freq domain: enforce |X|^2 to match target autocorrelation magnitude
            
            # Target |X|^2: we want R_target, so |X_target|^2 = fft(R_target)
            # But R is autocorrelation, so |X|^2 = fft(R)
            target_fft_R = fft(target_R, n=2*L)
            target_mag = np.abs(target_fft_R)
            
            # Current |X|
            current_mag = np.abs(X)
            
            # Blend: use target magnitude, keep current phase
            new_X = target_mag * (X / (current_mag + 1e-10))
            x_new = ifft(new_X, n=2*L).real[:L]
            
            # Enforce support: zero out indices > 128
            x_new[support_max+1:] = 0.0
            
            # Threshold to binary-like: keep top-k entries
            # We want roughly 15-35 elements
            k_target = int(np.sum(x_init > 0.5)) if np.sum(x_init > 0.5) > 0 else 25
            if k_target < 10:
                k_target = 25
            if k_target > 40:
                k_target = 40
            
            # Soft threshold: keep entries above a threshold
            # Use iterative thresholding
            for thresh_iter in range(5):
                thresh = np.sort(x_new)[::-1][min(k_target, len(x_new)-1)] * 0.9
                x_new = np.where(x_new > thresh, x_new, 0.0)
                # Re-normalize: scale so that sum of x^2 ~ k_target
                energy = np.sum(x_new**2)
                if energy > 0:
                    x_new *= np.sqrt(k_target / energy)
            
            x = x_new
        
        # Final binary threshold
        thresh = np.sort(x)[::-1][min(k_target, len(x)-1)] * 0.8
        x_bin = np.where(x > thresh, 1.0, 0.0)
        x_bin[support_max+1:] = 0.0
        
        # Ensure non-empty
        if np.sum(x_bin) == 0:
            x_bin[64] = 1.0
        
        return x_bin

    # Polish: exhaustive local search to refine
    def polish(A, sums, max_iters=200):
        """Greedy local search: try all single moves, keep best improvement each pass"""
        improved = True
        iters = 0
        current_cost = get_hard_cost(sums)
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_move = None
            best_new_cost = current_cost
            best_new_A = None
            best_new_sums = None
            
            # Try removing each element
            A_list = list(A)
            for x in A_list:
                if len(A) <= 10:
                    break
                # Apply remove
                apply_remove(A, sums, x)
                new_cost = get_hard_cost(sums)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_move = ('remove', x)
                    best_new_A = set(A)
                    best_new_sums = sums.copy()
                # Revert
                apply_add(A, sums, x)
            
            # Try adding each element
            if best_new_cost >= current_cost:
                candidates = [x for x in range(N+1) if x not in A]
                for x in candidates:
                    apply_add(A, sums, x)
                    new_cost = get_hard_cost(sums)
                    if new_cost < best_new_cost:
                        best_new_cost = new_cost
                        best_move = ('add', x)
                        best_new_A = set(A)
                        best_new_sums = sums.copy()
                    apply_remove(A, sums, x)
            
            # Try swaps
            if best_new_cost >= current_cost:
                A_list = list(A)
                for x in A_list:
                    candidates = [y for y in range(N+1) if y not in A]
                    for y in candidates:
                        apply_remove(A, sums, x)
                        apply_add(A, sums, y)
                        new_cost = get_hard_cost(sums)
                        if new_cost < best_new_cost:
                            best_new_cost = new_cost
                            best_move = ('swap', x, y)
                            best_new_A = set(A)
                            best_new_sums = sums.copy()
                        apply_remove(A, sums, y)
                        apply_add(A, sums, x)
            
            if best_move is not None:
                A = best_new_A
                sums = best_new_sums
                current_cost = best_new_cost
                improved = True
                
        return A, sums, current_cost

    # Initialize best solution
    best_A = None
    best_cost = float('inf')
    best_sums = None
    
    # Try multiple random initializations
    num_restarts = 15
    
    # Phase retrieval initialization
    def gs_init(seed_offset=0):
        """Generate initial set using Gerchberg-Saxton phase retrieval"""
        rng = random.Random(42 + seed_offset)
        x_init = np.zeros(257)
        # Random sparse initialization
        k = rng.randint(15, 35)
        indices = rng.sample(range(129), k)
        x_init[indices] = 1.0
        
        x_bin = gerchberg_saxton(x_init, n_iters=50, target_level=1.0, support_max=128)
        A = set(int(i) for i in range(129) if x_bin[i] > 0.5)
        if not A:
            A = set(range(64, 129))
        return A
    
    for init_idx in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Alternate between random and GS initializations
        if init_idx % 2 == 0:
            # GS phase retrieval initialization
            A = gs_init(init_idx)
        else:
            # Random set of size ~15-35
            size = random.randint(15, 35)
            A = set(random.sample(range(N+1), size))
        
        # Recalculate sums properly
        sums = np.zeros(257, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        
        current_cost = get_cost(sums)
        
        # Simulated Annealing with smooth cost
        T = 50.0
        T_min = 0.5
        cooling = 0.998
        steps = 25000
        
        current_A = A
        current_sums = sums
        current_cost = get_cost(current_sums)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(current_A)
            best_sums = current_sums.copy()
            
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Early termination if we found a perfect solution
            hard_cost = get_hard_cost(current_sums)
            if hard_cost < 1200:
                return sorted(current_A)

            # Choose move: 10% Add, 10% Remove, 80% Swap
            move_type = random.random()
            
            if move_type < 0.1:
                # Add
                candidates = [x for x in range(N+1) if x not in current_A]
                if not candidates: continue
                x = random.choice(candidates)
                
                apply_add(current_A, current_sums, x)
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    apply_remove(current_A, current_sums, x)
                    
            elif move_type < 0.2:
                # Remove
                if len(current_A) < 10: continue
                x = random.choice(list(current_A))
                
                apply_remove(current_A, current_sums, x)
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    apply_add(current_A, current_sums, x)
                    
            else:
                # Swap
                if len(current_A) == 0: continue
                x = random.choice(list(current_A))
                candidates = [y for y in range(N+1) if y not in current_A]
                if not candidates: continue
                y = random.choice(candidates)
                
                apply_remove(current_A, current_sums, x)
                apply_add(current_A, current_sums, y)
                
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    apply_remove(current_A, current_sums, y)
                    apply_add(current_A, current_sums, x)
            
            T *= cooling
            if T < T_min:
                break
        
        # Polish the best solution found in this restart
        if best_A is not None and time.time() < deadline - 15:
            polished_A, polished_sums, polished_cost = polish(best_A, best_sums.copy())
            if polished_cost < best_cost:
                best_cost = polished_cost
                best_A = polished_A
                best_sums = polished_sums
            
            # Periodic GS refinement: try GS on current best to find better starting point
            if time.time() < deadline - 8 and init_idx % 3 == 0:
                x_init = np.zeros(257)
                for a in best_A:
                    x_init[a] = 1.0
                x_gs = gerchberg_saxton(x_init, n_iters=40, target_level=1.0, support_max=128)
                A_gs = set(int(i) for i in range(129) if x_gs[i] > 0.5)
                if A_gs:
                    sums_gs = np.zeros(257, dtype=np.int32)
                    A_gs_list = list(A_gs)
                    for i in range(len(A_gs_list)):
                        for j in range(i, len(A_gs_list)):
                            s = A_gs_list[i] + A_gs_list[j]
                            if s <= 256:
                                sums_gs[s] += 1
                    cost_gs = get_hard_cost(sums_gs)
                    if cost_gs < best_cost:
                        best_cost = cost_gs
                        best_A = A_gs
                        best_sums = sums_gs.copy()

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return sorted(best_A)
