"""
Strategy: Multi-start Sign-Flip Local Search with Block Moves and Simulated Annealing.

The previous approach tried to optimize a continuous signal and quantize, which is
inefficient and often stuck in bad local minima. This approach searches the discrete
space of signed step functions directly.

Key improvements:
1. Discrete search: We only evaluate valid $\pm 1$ sequences.
2. Block Moves: Flipping a single sign is often insufficient to escape local minima.
   Flipping a contiguous block of signs allows for larger structural changes.
3. Simulated Annealing: Accepts worse moves with a decaying probability to escape
   local traps.
4. Multi-start: We run several independent annealing schedules from different random
   initializations to increase the chance of finding a global optimum.
5. Efficient Objective: Uses FFT for autoconvolution.
"""

import numpy as np
import time

N = 400

def _compute_objective(f):
    """
    Compute max |autocorr| / (sum f)^2.
    f is a numpy array of +/- 1.
    """
    n = len(f)
    # Autoconvolution via FFT
    # We need linear autoconvolution.
    # A[k] = sum_{i} f[i] f[i+k]
    # Using FFT of length 2n (or next power of 2 for speed, but 2n is fine for N=400)
    f_hat = np.fft.fft(f, n=2*n)
    A = np.fft.ifft(f_hat * np.conj(f_hat)).real
    # The first n elements of A correspond to lags 0 to n-1.
    # For real f, A is symmetric, so max |A[0:n]| is the max absolute autoconv value.
    A = A[:n]
    
    max_abs_ac = np.max(np.abs(A))
    integral = np.sum(f)
    if integral == 0:
        return np.inf
    return max_abs_ac / (integral ** 2)

def _anneal(f_init, time_budget):
    """
    Run simulated annealing on a sign sequence.
    """
    f = f_init.copy()
    current_obj = _compute_objective(f)
    best_f = f.copy()
    best_obj = current_obj
    
    # Initial temperature
    # The objective is roughly O(1/N) or similar.
    # Let's estimate the scale of changes.
    # A flip changes sum by 2.
    # Let's start T such that we accept moves that worsen obj by ~10% initially.
    # We need to be careful with T scaling.
    # Let's normalize objective by N^2? No, obj is already normalized by (sum)^2 ~ N^2.
    # So obj is small.
    # Let's just use a heuristic T.
    T = 0.05 
    T_min = 1e-6
    cooling_rate = 0.95
    
    start_time = time.time()
    
    # We will try block flips of size 1 to 5.
    block_sizes = [1, 2, 3, 4, 5]
    
    while (time.time() - start_time) < time_budget:
        if T < T_min:
            break
            
        # Select a move
        # Random start index and block size
        b_size = np.random.choice(block_sizes, p=[0.4, 0.2, 0.15, 0.1, 0.15])
        start_idx = np.random.randint(0, N - b_size + 1)
        
        # Create new candidate
        f_new = f.copy()
        f_new[start_idx:start_idx+b_size] *= -1
        
        new_obj = _compute_objective(f_new)
        
        # Simulated Annealing Acceptance
        delta_obj = new_obj - current_obj
        if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
            f = f_new
            current_obj = new_obj
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_f = f.copy()
        
        T *= cooling_rate
        
    return best_f, best_obj

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s for safety
    
    best_f = None
    best_obj = np.inf
    
    # Strategy: Multi-start annealing
    # We will run several independent annealing runs.
    # How many? Depends on time.
    # Each annealing run takes ~ (time_budget)
    # Let's allocate time per run and see how many we can fit.
    
    # First, try a few deterministic structured starts
    # 1. All ones (bad, but baseline)
    # 2. Alternating 1, -1 (Good for low autoconv?)
    # 3. Random
    
    candidates = []
    
    # Start 1: Random
    np.random.seed(1)
    f_rand = np.random.choice([-1, 1], size=N)
    candidates.append(f_rand)
    
    # Start 2: Alternating
    f_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(f_alt)
    
    # Start 3: Block pattern (e.g., 10 ones, 10 minus ones)
    f_block = np.tile([1.0]*10 + [-1.0]*10, N//20)
    if len(f_block) < N:
        f_block = np.pad(f_block, (0, N-len(f_block)), mode='edge')
    candidates.append(f_block)
    
    # Start 4: Random with seed 42
    np.random.seed(42)
    f_rand2 = np.random.choice([-1, 1], size=N)
    candidates.append(f_rand2)
    
    # Evaluate initial candidates
    for f in candidates:
        obj = _compute_objective(f)
        if obj < best_obj:
            best_obj = obj
            best_f = f.copy()
            
    # Now run annealing on the best initial candidate and a few others
    # We have ~130s left.
    # Let's do 5 annealing runs of 20s each? Or 10 runs of 10s?
    # Annealing needs time to cool. 20s is reasonable.
    
    num_runs = 5
    time_per_run = 20.0
    
    # We will anneal from:
    # 1. The best initial candidate
    # 2. The second best initial candidate
    # 3. A new random start
    # 4. A new random start
    # 5. A new random start
    
    starts_for_annealing = [best_f.copy()]
    
    # Add second best
    objs = []
    for f in candidates:
        objs.append(_compute_objective(f))
    second_best_idx = np.argsort(objs)[1]
    starts_for_annealing.append(candidates[second_best_idx].copy())
    
    # Add 3 new randoms
    for i in range(3):
        np.random.seed(100 + i)
        f_new_rand = np.random.choice([-1, 1], size=N)
        starts_for_annealing.append(f_new_rand)
    
    for i, f_start in enumerate(starts_for_annealing):
        if time.time() - start_time > deadline:
            break
            
        # Check if we have enough time for this run
        if time.time() - start_time + time_per_run > deadline:
            time_per_run = deadline - (time.time() - start_time)
            if time_per_run < 5.0:
                break
                
        f_res, obj_res = _anneal(f_start, time_per_run)
        
        if obj_res < best_obj:
            best_obj = obj_res
            best_f = f_res.copy()
            
    # If we have time left, do a final polish with single-flip local search on best_f
    if best_f is not None and time.time() - start_time < deadline - 10:
        # Simple local search polish
        f_polish = best_f.copy()
        obj_polish = best_obj
        improved = True
        polish_deadline = deadline - 5
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                f_try = f_polish.copy()
                f_try[i] *= -1
                obj_try = _compute_objective(f_try)
                if obj_try < obj_polish:
                    obj_polish = obj_try
                    f_polish = f_try
                    improved = True
                    if obj_polish < best_obj:
                        best_obj = obj_polish
                        best_f = f_polish.copy()
                        
    return best_f

# EVOLVE-BLOCK-END
