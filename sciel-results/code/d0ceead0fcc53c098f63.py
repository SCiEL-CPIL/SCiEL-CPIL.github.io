import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Target: every t in [128, 256] must be expressible as a+b with a<=b, a,b in A
    # Minimize max multiplicity over t in [128, 256]
    
    # Precompute FFT size for fast convolution
    fft_size = 512  # power of 2 >= 257
    
    def fast_convolve(v):
        """Compute convolution v*v using FFT."""
        V = np.fft.rfft(v, n=fft_size)
        conv = np.fft.irfft(V * V, n=fft_size)
        return np.round(conv).astype(np.int32)
    
    def compute_unordered_fast(conv, A_vec):
        """Compute unordered representation counts from convolution and vector."""
        ordered = conv[128:257].astype(np.int32)
        t_vals = np.arange(128, 257)
        is_even = t_vals % 2 == 0
        mids = t_vals // 2
        diag_contrib = (is_even & (A_vec[mids] > 0)).astype(np.int32)
        unordered = (ordered + diag_contrib) // 2
        return unordered
    
    def evaluate_vector(A_vec):
        """Fast evaluation using pre-allocated vector. Returns (max_mult, covered, target)."""
        conv = fast_convolve(A_vec)
        target = compute_unordered_fast(conv, A_vec)
        max_mult = int(target.max())
        covered = bool(target.min() >= 1)
        return max_mult, covered, target
    
    def build_initial_set_v2():
        """Build a valid covering set using structured approach."""
        # Start with all even numbers - this covers all even targets
        A = set(range(0, 129, 2))
        
        # Add some odd numbers strategically to cover odd targets
        # For odd t in [128,256], need one even and one odd from A
        # Add odd numbers that help cover many targets
        odd_candidates = list(range(1, 129, 2))
        random.shuffle(odd_candidates)
        
        for x in odd_candidates:
            if len(A) > 100:
                break
            A.add(x)
            # Check if we have good coverage
            vec = np.zeros(129, dtype=np.float64)
            for a in A:
                vec[a] = 1.0
            _, covered, _ = evaluate_vector(vec)
            if covered:
                break
        
        return A
    
    A0 = build_initial_set_v2()
    score0, covered0, target0 = evaluate_vector(np.array([1.0 if x in A0 else 0.0 for x in range(129)]))
    
    if not covered0:
        # Fallback to denser set
        A0 = set(range(0, 129, 2)) | set(range(1, 129, 4))
        vec0 = np.zeros(129, dtype=np.float64)
        for x in A0:
            vec0[x] = 1.0
        score0, covered0, target0 = evaluate_vector(vec0)
    
    if not covered0:
        A0 = set(range(0, 129))
        vec0 = np.zeros(129, dtype=np.float64)
        for x in A0:
            vec0[x] = 1.0
        score0, covered0, target0 = evaluate_vector(vec0)
    
    best_A = set(A0)
    best_score = score0
    best_covered = covered0
    
    # Convert to vector for fast SA
    current_vec = np.zeros(129, dtype=np.float64)
    for x in A0:
        current_vec[x] = 1.0
    
    current_score, current_covered, current_target = evaluate_vector(current_vec)
    
    # Adaptive simulated annealing with multiple restarts
    T = 8.0
    T_min = 0.005
    cooling = 0.9997
    
    all_indices = np.arange(129)
    
    # Track best solutions found
    best_solutions = []
    
    while time.time() < deadline - 10 and T > T_min:
        r = random.random()
        new_vec = current_vec.copy()
        
        if r < 0.35 and current_vec.sum() > 2:
            # Remove an element
            idxs = np.nonzero(current_vec)[0]
            if len(idxs) == 0: continue
            x = idxs[random.randint(0, len(idxs)-1)]
            new_vec[x] = 0.0
        elif r < 0.65:
            # Add an element
            idxs = np.nonzero(1.0 - current_vec)[0]
            if len(idxs) == 0: continue
            x = idxs[random.randint(0, len(idxs)-1)]
            new_vec[x] = 1.0
        else:
            # Swap: remove one, add another
            idxs_in = np.nonzero(current_vec)[0]
            idxs_out = np.nonzero(1.0 - current_vec)[0]
            if len(idxs_in) == 0 or len(idxs_out) == 0: continue
            x_in = idxs_in[random.randint(0, len(idxs_in)-1)]
            x_out = idxs_out[random.randint(0, len(idxs_out)-1)]
            new_vec[x_in] = 0.0
            new_vec[x_out] = 1.0
        
        new_score, new_covered, new_target = evaluate_vector(new_vec)
        
        # Use a smooth objective: penalize uncovered targets heavily
        new_cost = new_score + (1000 if not new_covered else 0)
        current_cost = current_score + (1000 if not current_covered else 0)
        
        delta = new_cost - current_cost
        
        # Accept if better, or with probability based on temperature
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_vec = new_vec
            current_score = new_score
            current_covered = new_covered
            current_target = new_target
            
            if current_covered and current_score < best_score:
                best_score = current_score
                best_covered = True
                best_A = set(np.nonzero(current_vec)[0].tolist())
                best_solutions.append((best_score, set(best_A)))
        
        T *= cooling
        
        # Occasionally restart from best solution to escape local minima
        if len(best_solutions) > 0 and random.random() < 0.01:
            best_sol_score, best_sol_set = random.choice(best_solutions[-5:])
            current_vec = np.zeros(129, dtype=np.float64)
            for x in best_sol_set:
                current_vec[x] = 1.0
            current_score, current_covered, current_target = evaluate_vector(current_vec)
            T = max(T, 2.0)  # Reset temperature
    
    # Final polish: aggressive local search
    if best_covered and time.time() < deadline - 2:
        # Multi-pass removal
        for _ in range(3):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for x in sorted(best_A):
                    if time.time() > deadline - 2:
                        break
                    new_A = best_A - {x}
                    new_vec = np.zeros(129, dtype=np.float64)
                    for a in new_A:
                        new_vec[a] = 1.0
                    s, c, _ = evaluate_vector(new_vec)
                    if c and s <= best_score:
                        best_A = new_A
                        best_score = s
                        improved = True
        
        # Try swaps to reduce max multiplicity
        if time.time() < deadline - 1:
            for _ in range(200):
                if time.time() > deadline - 1:
                    break
                if not best_A:
                    break
                x_in = random.choice(sorted(best_A))
                idxs_out = np.nonzero(1.0 - np.array([1.0 if x in best_A else 0.0 for x in range(129)]))[0]
                if len(idxs_out) == 0:
                    break
                x_out = idxs_out[random.randint(0, len(idxs_out)-1)]
                new_A = best_A - {x_in} | {x_out}
                new_vec = np.zeros(129, dtype=np.float64)
                for a in new_A:
                    new_vec[a] = 1.0
                s, c, _ = evaluate_vector(new_vec)
                if c and s < best_score:
                    best_A = new_A
                    best_score = s
    
    # Ensure validity
    if not best_covered:
        best_A = set(range(0, 129))
    
    return sorted(best_A)
