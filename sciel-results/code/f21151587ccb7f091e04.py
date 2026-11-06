"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
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
    
    # Build the AP matrix as a numpy array of shape (num_aps, N) with int8
    M = np.zeros((num_aps, N), dtype=np.int8)
    for i, ap in enumerate(aps):
        for idx in ap:
            M[i, idx] = 1
    
    # Convert to float32 for fast matvec
    Mf = M.astype(np.float32)
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute sum over all APs using matrix-vector product."""
        sums = Mf.dot(signs.astype(np.float32))
        return int(np.max(np.abs(sums)))
    
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n: t(i) = (-1)^(popcount(i))."""
        result = np.ones(n, dtype=np.int8)
        for i in range(1, n):
            # t(i) = -t(i & -i) * t(i - (i & -i))
            # Simpler: t(i) = (-1)^{number of 1s in binary representation of i}
            # Use parity of popcount
            x = i
            parity = 0
            while x:
                parity ^= (x & 1)
                x >>= 1
            result[i] = -1 if parity else 1
        return result
    
    def greedy_coordinate_descent(signs, max_iterations=100):
        """Greedy coordinate descent: flip each position if it reduces max abs AP sum."""
        signs = signs.copy()
        current_cost = compute_max_abs_sum(signs)
        
        for iteration in range(max_iterations):
            if time.time() >= deadline - 5.0:
                break
            improved = False
            for idx in range(N):
                if time.time() >= deadline - 5.0:
                    break
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_cost = compute_max_abs_sum(signs)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    signs[idx] = old_sign
            if not improved:
                break
        
        return signs, current_cost
    
    def simulated_annealing(signs, initial_temp=3.0, cooling_rate=0.999, min_temp=0.01):
        """Simulated annealing with single-flip moves."""
        signs = signs.copy()
        current_cost = compute_max_abs_sum(signs)
        best_cost = current_cost
        best_signs = signs.copy()
        T = initial_temp
        rng = random.Random(123)
        
        while time.time() < deadline - 10.0 and T > min_temp:
            idx = rng.randrange(N)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            new_cost = compute_max_abs_sum(signs)
            
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_signs = signs.copy()
            else:
                signs[idx] = old_sign
            
            T *= cooling_rate
        
        return best_signs, best_cost
    
    # Phase 1: Thue-Morse initialization + greedy descent
    best_signs = None
    best_cost = float('inf')
    
    # Thue-Morse start
    tm_signs = thue_morse(N)
    tm_signs, tm_cost = greedy_coordinate_descent(tm_signs, max_iterations=50)
    if tm_cost < best_cost:
        best_cost = tm_cost
        best_signs = tm_signs.copy()
    
    # Try a few random initializations with greedy descent
    for seed in [42, 137, 256, 512]:
        if time.time() >= deadline - 15.0:
            break
        rng_init = random.Random(seed)
        rand_signs = np.array([rng_init.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        rand_signs, rand_cost = greedy_coordinate_descent(rand_signs, max_iterations=30)
        if rand_cost < best_cost:
            best_cost = rand_cost
            best_signs = rand_signs.copy()
    
    # Phase 2: Simulated annealing refinement from best found
    if best_signs is not None:
        sa_signs, sa_cost = simulated_annealing(best_signs, initial_temp=2.0, cooling_rate=0.9995, min_temp=0.05)
        if sa_cost < best_cost:
            best_cost = sa_cost
            best_signs = sa_signs.copy()
    
    # Phase 3: Final greedy polish
    if best_signs is not None:
        final_signs, final_cost = greedy_coordinate_descent(best_signs, max_iterations=20)
        if final_cost < best_cost:
            best_cost = final_cost
            best_signs = final_signs.copy()
    
    if best_signs is None:
        # Fallback: Thue-Morse only
        best_signs = thue_morse(N)
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
