import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _get_cost(h):
    """
    Compute the upper bound on C3: max(|autocorr_sidelobe|)
    """
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:N]
    # C[0] is the DC component
    # We want max absolute value of sidelobes C[1:]
    if N > 1:
        max_side = np.max(np.abs(C[1:]))
    else:
        max_side = 0.0
    return max_side

def _local_search(h, max_iters=100, max_time=10.0):
    """
    Greedy local search with single-bit flips and small block flips.
    """
    start_time = time.time()
    current_cost = _get_cost(h)
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Try single flips in random order
        indices = np.random.permutation(N)
        for i in indices:
            if (time.time() - start_time) > max_time:
                break
            h[i] *= -1
            new_cost = _get_cost(h)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                h[i] *= -1
        
        # Try small block flips (length 2-4)
        for block_len in [2, 3, 4]:
            for _ in range(N // block_len):
                if (time.time() - start_time) > max_time:
                    break
                start_pos = np.random.randint(0, N - block_len + 1)
                h[start_pos:start_pos + block_len] *= -1
                new_cost = _get_cost(h)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    h[start_pos:start_pos + block_len] *= -1
    return h, current_cost

def _simulated_annealing(h, T_start=10.0, T_end=0.01, iterations=5000, max_time=20.0):
    """
    Simulated annealing with block moves.
    """
    start_time = time.time()
    current_cost = _get_cost(h)
    best_h = h.copy()
    best_cost = current_cost
    T = T_start
    
    for iteration in range(iterations):
        if (time.time() - start_time) > max_time:
            break
        
        # Choose random move: single flip or block flip
        if np.random.random() < 0.3:
            # Single flip
            idx = np.random.randint(N)
            h[idx] *= -1
            move_type = 'single'
            move_params = (idx,)
        else:
            # Block flip
            block_len = np.random.choice([2, 4, 8, 16, 32])
            if block_len >= N:
                block_len = N // 2
            start_pos = np.random.randint(0, N - block_len + 1)
            h[start_pos:start_pos + block_len] *= -1
            move_type = 'block'
            move_params = (start_pos, block_len)
        
        new_cost = _get_cost(h)
        delta = new_cost - current_cost
        
        # Accept or reject
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_h = h.copy()
        else:
            # Revert
            if move_type == 'single':
                h[move_params[0]] *= -1
            else:
                h[move_params[0]:move_params[0] + move_params[1]] *= -1
        
        # Cooling
        T = T_start * (T_end / T_start) ** (iteration / iterations)
    
    return best_h, best_cost

def construct():
    deadline = time.time() + 110.0
    np.random.seed(42)
    
    # Start with a random sequence
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_cost = _get_cost(best_h)
    
    # Phase 1: Multiple SA restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Initialize with random or perturbed best
        if restart == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        else:
            h = best_h.copy()
            # Perturb with random flips
            num_flips = np.random.randint(N // 10, N // 2)
            flip_indices = np.random.choice(N, num_flips, replace=False)
            h[flip_indices] *= -1
        
        # Run SA
        sa_h, sa_cost = _simulated_annealing(
            h, 
            T_start=5.0 + restart * 0.5, 
            T_end=0.01, 
            iterations=3000,
            max_time=min(15.0, deadline - time.time() - 15.0)
        )
        
        if sa_cost < best_cost:
            best_cost = sa_cost
            best_h = sa_h.copy()
        
        # Local search polish on SA result
        if time.time() < deadline - 10.0:
            polished_h, polished_cost = _local_search(sa_h.copy(), max_iters=50, max_time=min(5.0, deadline - time.time() - 8.0))
            if polished_cost < best_cost:
                best_cost = polished_cost
                best_h = polished_h.copy()
    
    # Phase 2: Intensive local search on best
    if time.time() < deadline - 5.0:
        final_h, final_cost = _local_search(
            best_h.copy(), 
            max_iters=200, 
            max_time=min(10.0, deadline - time.time() - 3.0)
        )
        if final_cost < best_cost:
            best_h = final_h
            best_cost = final_cost
    
    return best_h
# EVOLVE-BLOCK-END
