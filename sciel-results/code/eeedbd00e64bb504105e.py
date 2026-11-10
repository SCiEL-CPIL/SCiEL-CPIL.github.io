import time
import random
import numpy as np

def construct():
    # Deadline: 85 seconds to be safe (evaluator kills at 90s)
    deadline = time.time() + 85.0
    
    # Target sums: 128 to 256 inclusive
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    
    # Precompute: for each x in [0,128], the list of sum indices it contributes to
    # when paired with element a. Actually, we need: for a given x, which sums does
    # x contribute to? It contributes to sums x+a for each a in A.
    # But for incremental updates, we need: when removing x, for each a in A,
    # decrement M[x+a-MIN_SUM] if in range.
    
    # Precompute partner indices: for each x, the array of (a, sum_index) pairs
    # where sum_index = x+a-MIN_SUM is valid (0 <= sum_index < NUM_SUMS)
    # But this depends on A. Instead, precompute for each x, the mapping from a to sum_index
    
    # For efficiency, precompute: for each x in [0,128], 
    #   partner_sum_idx[x][a] = x+a-MIN_SUM if MIN_SUM <= x+a <= MAX_SUM, else -1
    # This is a 129x129 int array
    partner_sum_idx = np.full((129, 129), -1, dtype=np.int32)
    for x in range(129):
        for a in range(129):
            s = x + a
            if MIN_SUM <= s <= MAX_SUM:
                partner_sum_idx[x, a] = s - MIN_SUM
    
    # Helper to calculate multiplicities for a given set A
    def calc_mult(A):
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        A_list = sorted(A)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            # a + a
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            # a + b for b > a
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if MIN_SUM <= s <= MAX_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    # Helper: remove element x from A and update M incrementally
    def remove_from_A(x, A, M):
        """Remove x from A, update M. Returns (new_A, new_M) or (None, None) if invalid."""
        new_M = M.copy()
        # For each a in A: decrement M[x+a - MIN_SUM] if in range
        for a in A:
            idx = partner_sum_idx[x, a]
            if idx >= 0:
                new_M[idx] -= 1
        # Check validity
        if np.min(new_M) < 1:
            return None, None
        new_A = A - {x}
        return new_A, new_M
    
    # Helper: add element x to A and update M incrementally
    def add_to_A(x, A, M):
        """Add x to A, update M. Returns (new_A, new_M)."""
        new_M = M.copy()
        # For each a in A: increment M[x+a - MIN_SUM] if in range
        for a in A:
            idx = partner_sum_idx[x, a]
            if idx >= 0:
                new_M[idx] += 1
        # x + x
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            new_M[s - MIN_SUM] += 1
        new_A = A | {x}
        return new_A, new_M
    
    # Helper: compute a smooth surrogate for the peak using log-sum-exp
    def smooth_peak(M, beta=8.0):
        """Compute a smooth approximation of max(M) using log-sum-exp."""
        M_float = M.astype(np.float64)
        max_M = np.max(M_float)
        if max_M == 0:
            return 0.0
        shifted = M_float - max_M
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_M + log_sum / beta
    
    # Better initial sets
    initial_sets = [
        set(range(64, 129)),
        set(range(48, 113)),
        set(range(32, 97)),
        set(range(16, 81)),
        set(range(0, 129, 2)),
        set(range(0, 129, 3)),
        set(range(64, 129)) | set(range(0, 32, 2)),
        set(range(48, 113)) | set(range(0, 16, 2)),
    ]
    
    best_A = None
    best_score = float('inf')
    best_M = None
    
    for init_set in initial_sets:
        M_init = calc_mult(init_set)
        if np.min(M_init) < 1:
            continue
        
        A = set(init_set)
        M = M_init
        current_score = int(np.max(M))
        
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
            best_M = M.copy()
    
    if best_A is None:
        A = set(range(64, 129))
        M = calc_mult(A)
        best_A = set(A)
        best_M = M.copy()
        best_score = int(np.max(M))
    
    # Simulated Annealing with improved moves
    ALL_ELEMENTS = list(range(0, 129))
    ALL_ELEMENTS_SET = set(ALL_ELEMENTS)
    
    # We'll do multiple restarts to escape local optima
    max_restarts = 15
    restart_time = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize for this restart
        if restart == 0:
            A = set(best_A)
            M = best_M.copy()
            current_score = best_score
        else:
            # Start from a random perturbation of best_A
            A = set(best_A)
            M = best_M.copy()
            current_score = best_score
            
            # Perturb: randomly remove and add some elements
            num_perturb = random.randint(5, 15)
            for _ in range(num_perturb):
                if len(A) > 5:
                    x = random.choice(list(A))
                    new_A, new_M = remove_from_A(x, A, M)
                    if new_M is not None:
                        A = new_A
                        M = new_M
                        current_score = int(np.max(M))
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if candidates:
                    y = random.choice(candidates)
                    new_A, new_M = add_to_A(y, A, M)
                    A = new_A
                    M = new_M
                    current_score = int(np.max(M))
        
        T = 4.0
        cooling = 0.9995
        beta = 8.0  # For smooth peak
        
        # Track smooth score for acceptance
        current_smooth = smooth_peak(M, beta)
        
        # Limit iterations per restart
        max_iters = 80000
        iters = 0
        
        while time.time() < deadline and iters < max_iters:
            iters += 1
            
            # Choose a move type with bias toward removal/swap
            move_type = random.random()
            
            if move_type < 0.35:
                # Remove an element
                if len(A) <= 8:
                    T *= cooling
                    continue
                
                # Prefer removing elements that contribute to high-multiplicity sums
                A_list = list(A)
                x = random.choice(A_list)
                
                new_A, new_M = remove_from_A(x, A, M)
                if new_M is not None:
                    score_new = int(np.max(new_M))
                    delta = score_new - current_score
                    
                    # Use smooth surrogate for acceptance
                    smooth_new = smooth_peak(new_M, beta)
                    delta_smooth = smooth_new - current_smooth
                    
                    if delta < 0 or random.random() < np.exp(-delta_smooth / max(T, 0.01)):
                        A = new_A
                        M = new_M
                        current_score = score_new
                        current_smooth = smooth_new
                        if score_new < best_score:
                            best_score = score_new
                            best_A = set(A)
                            best_M = M.copy()
            
            elif move_type < 0.75:
                # Swap: remove x, add y
                if len(A) == 0:
                    T *= cooling
                    continue
                
                x = random.choice(list(A))
                # Choose y not in A
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if not candidates:
                    T *= cooling
                    continue
                y = random.choice(candidates)
                
                # Simulate: remove x then add y
                temp_A, temp_M = remove_from_A(x, A, M)
                if temp_M is not None:
                    new_A, new_M = add_to_A(y, temp_A, temp_M)
                    score_new = int(np.max(new_M))
                    delta = score_new - current_score
                    
                    # Use smooth surrogate for acceptance
                    smooth_new = smooth_peak(new_M, beta)
                    delta_smooth = smooth_new - current_smooth
                    
                    if delta < 0 or random.random() < np.exp(-delta_smooth / max(T, 0.01)):
                        A = new_A
                        M = new_M
                        current_score = score_new
                        current_smooth = smooth_new
                        if score_new < best_score:
                            best_score = score_new
                            best_A = set(A)
                            best_M = M.copy()
            
            else:
                # Add an element (less common)
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if not candidates:
                    T *= cooling
                    continue
                y = random.choice(candidates)
                
                new_A, new_M = add_to_A(y, A, M)
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                
                # Use smooth surrogate for acceptance
                smooth_new = smooth_peak(new_M, beta)
                delta_smooth = smooth_new - current_smooth
                
                # Adding usually increases max, so only accept with low probability
                if delta <= 0 or random.random() < np.exp(-delta_smooth / max(T, 0.01)) * 0.5:
                    A = new_A
                    M = new_M
                    current_score = score_new
                    current_smooth = smooth_new
            
            # Cool down
            T *= cooling
            if T < 0.03:
                T = 0.03
        
        # Final polish for this restart: try to remove more elements
        if time.time() < deadline:
            removed = True
            while removed and time.time() < deadline:
                removed = False
                A_list = list(A)
                random.shuffle(A_list)
                for x in A_list:
                    if x not in A:
                        continue
                    new_A, new_M = remove_from_A(x, A, M)
                    if new_M is not None:
                        score_new = int(np.max(new_M))
                        if score_new <= best_score:
                            A = new_A
                            M = new_M
                            current_score = score_new
                            if score_new < best_score:
                                best_score = score_new
                                best_A = set(A)
                                best_M = M.copy()
                            removed = True
    
    # Final global polish: try to remove elements from best_A
    if time.time() < deadline - 2:
        A = set(best_A)
        M = best_M.copy()
        current_score = best_score
        removed = True
        while removed and time.time() < deadline - 2:
            removed = False
            A_list = list(A)
            random.shuffle(A_list)
            for x in A_list:
                if x not in A:
                    continue
                new_A, new_M = remove_from_A(x, A, M)
                if new_M is not None:
                    score_new = int(np.max(new_M))
                    if score_new <= best_score:
                        A = new_A
                        M = new_M
                        current_score = score_new
                        if score_new < best_score:
                            best_score = score_new
                            best_A = set(A)
                            best_M = M.copy()
                        removed = True
    
    return list(best_A)
