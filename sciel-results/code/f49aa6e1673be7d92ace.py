import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n  # 256
    N_fft = 512  # Next power of 2 >= 257

    # Precompute FFT weights
    def compute_counts_fft(x):
        """
        x: boolean numpy array of length N_fft (1 if element in A, 0 otherwise)
        Returns numpy array of length max_sum+1 where counts[s] = number of unordered pairs (a,b) in A with a<=b and a+b=s
        """
        X = np.fft.rfft(x, N_fft)
        corr = np.fft.irfft(X * np.conjugate(X), N_fft)
        # corr[s] for s in 0..256 gives ordered count (a,b) where a+b=s mod N_fft
        # For s <= 256, since N_fft=512 > 256, no wraparound
        # Ordered count: for a != b, counted twice; for a == b, counted once
        # Unordered count = (ordered_count + diagonal) / 2
        # diagonal[s] = 1 if s is even and s//2 is in A, else 0
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for s in range(target_lo, target_hi + 1):
            ordered = int(round(corr[s]))
            diag = 1 if (s % 2 == 0 and x[s // 2] == 1) else 0
            counts[s] = (ordered + diag) // 2
        return counts

    def compute_full_counts_fft(x):
        """Compute counts for all sums 0..256."""
        X = np.fft.rfft(x, N_fft)
        corr = np.fft.irfft(X * np.conjugate(X), N_fft)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for s in range(max_sum + 1):
            ordered = int(round(corr[s]))
            diag = 1 if (s % 2 == 0 and x[s // 2] == 1) else 0
            counts[s] = (ordered + diag) // 2
        return counts

    def get_score(counts):
        """Returns (uncovered, peak, sum_sq) for target range."""
        target_counts = counts[target_lo:target_hi + 1]
        uncovered = np.sum(target_counts == 0)
        peak = np.max(target_counts)
        sum_sq = np.sum(target_counts * target_counts)
        return (uncovered, peak, sum_sq)

    def init_x(A):
        """Convert set A to numpy boolean vector."""
        x = np.zeros(N_fft, dtype=np.int8)
        for a in A:
            x[a] = 1
        return x

    def x_to_set(x):
        return set(np.where(x == 1)[0].tolist())

    # --- Fast local search using FFT ---
    def fast_local_search(A_init, max_steps=200000):
        """
        Stochastic local search with FFT-based evaluation.
        Moves: add, remove, swap.
        Objective: lexicographic (uncovered, peak, sum_sq)
        """
        A = set(A_init)
        x = init_x(A)
        counts = compute_full_counts_fft(x)
        curr_score = get_score(counts)
        
        best_A = set(A)
        best_score = curr_score
        best_x = x.copy()
        
        # Tabu-like: avoid immediate undoing
        last_move = None  # (type, arg1, arg2)
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            
            uncovered, peak, sum_sq = curr_score
            
            # Choose move type
            r = random.random()
            if uncovered > 0:
                # Prioritize adding elements to cover gaps
                if r < 0.5:
                    move_type = 'add'
                elif r < 0.9:
                    move_type = 'swap'
                else:
                    move_type = 'remove'
            else:
                # Full coverage: focus on reducing peak
                if r < 0.1:
                    move_type = 'add'
                elif r < 0.7:
                    move_type = 'swap'
                else:
                    move_type = 'remove'
            
            # Generate and evaluate move
            accepted = False
            new_x = x.copy()
            
            if move_type == 'add':
                if len(A) < n + 1:
                    candidates = [i for i in range(n + 1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        new_x[y] = 1
                        new_counts = compute_full_counts_fft(new_x)
                        new_score = get_score(new_counts)
                        
                        # Accept if better or equal uncovered with lower peak
                        if new_score < curr_score:
                            x = new_x
                            A.add(y)
                            counts = new_counts
                            curr_score = new_score
                            last_move = ('add', y, None)
                            if curr_score < best_score:
                                best_score = curr_score
                                best_A = set(A)
                                best_x = x.copy()
                            accepted = True
                            
            elif move_type == 'swap':
                if len(A) > 0:
                    x_elem = random.choice(list(A))
                    candidates = [i for i in range(n + 1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        new_x[x_elem] = 0
                        new_x[y] = 1
                        new_counts = compute_full_counts_fft(new_x)
                        new_score = get_score(new_counts)
                        
                        if new_score < curr_score:
                            x = new_x
                            A.remove(x_elem)
                            A.add(y)
                            counts = new_counts
                            curr_score = new_score
                            last_move = ('swap', x_elem, y)
                            if curr_score < best_score:
                                best_score = curr_score
                                best_A = set(A)
                                best_x = x.copy()
                            accepted = True
                            
            else:  # remove
                if len(A) > 1:
                    x_elem = random.choice(list(A))
                    new_x[x_elem] = 0
                    new_counts = compute_full_counts_fft(new_x)
                    new_score = get_score(new_counts)
                    
                    if new_score < curr_score:
                        x = new_x
                        A.remove(x_elem)
                        counts = new_counts
                        curr_score = new_score
                        last_move = ('remove', x_elem, None)
                        if curr_score < best_score:
                            best_score = curr_score
                            best_A = set(A)
                            best_x = x.copy()
                        accepted = True
            
            # Simulated annealing acceptance for non-improving moves (only when covered)
            if not accepted and uncovered == 0 and step % 10 == 0:
                # Occasionally accept a slightly worse move to escape local optima
                T = max(0.5, 5.0 - step / max_steps * 4.5)
                r2 = random.random()
                if r2 < 0.15:
                    # Try a random swap and accept with probability based on cost difference
                    if len(A) > 0:
                        x_elem = random.choice(list(A))
                        candidates = [i for i in range(n + 1) if i not in A]
                        if candidates:
                            y = random.choice(candidates)
                            new_x[x_elem] = 0
                            new_x[y] = 1
                            new_counts = compute_full_counts_fft(new_x)
                            new_score = get_score(new_counts)
                            
                            old_cost = curr_score[0] * 1000 + curr_score[1] * 10 + curr_score[2] / 100.0
                            new_cost = new_score[0] * 1000 + new_score[1] * 10 + new_score[2] / 100.0
                            d_cost = new_cost - old_cost
                            
                            if d_cost < 0:
                                accept = True
                            else:
                                accept = (random.random() < np.exp(-d_cost / T))
                            
                            if accept:
                                x = new_x
                                A.remove(x_elem)
                                A.add(y)
                                counts = new_counts
                                curr_score = new_score
                                if curr_score < best_score:
                                    best_score = curr_score
                                    best_A = set(A)
                                    best_x = x.copy()
        
        return best_A, best_score

    # --- Initial set generation ---
    def generate_initial_sets():
        """Generate diverse initial sets."""
        inits = []
        
        # 1. Arithmetic progressions
        for step in [2, 3, 4]:
            for start in range(step):
                A_ap = set(range(start, n + 1, step))
                inits.append(A_ap)
        
        # 2. Structured: split into two interleaved APs
        for step in [2, 3]:
            A_half1 = set(range(0, n + 1, 2 * step))
            A_half2 = set(range(step, n + 1, 2 * step))
            inits.append(A_half1 | A_half2)  # Same as range(0, n+1, step)
            inits.append(A_half1)
            inits.append(A_half2)
        
        # 3. Random sets of various sizes
        for size in [30, 40, 50, 60, 70]:
            for _ in range(3):
                A_rand = set(random.sample(range(n + 1), size))
                inits.append(A_rand)
        
        # 4. Symmetric sets (if a in A then 256-a in A, clipped to [0,128])
        for _ in range(5):
            base = set(random.sample(range(0, 65), 20))
            A_sym = set()
            for a in base:
                A_sym.add(a)
                if 256 - a <= n:
                    A_sym.add(256 - a)
            inits.append(A_sym)
        
        # 5. Greedy coverage: start with elements that cover many target sums
        A_greedy = set()
        # Add 0 and 128 first (extremes help cover 128 and 256)
        A_greedy.add(0)
        A_greedy.add(128)
        # Greedily add element that covers most uncovered sums
        x_g = init_x(A_greedy)
        counts_g = compute_full_counts_fft(x_g)
        for _ in range(50):
            uncovered_sums = [s for s in range(target_lo, target_hi + 1) if counts_g[s] == 0]
            if not uncovered_sums:
                break
            best_y = None
            best_cov = 0
            for y in range(n + 1):
                if y in A_greedy:
                    continue
                # How many uncovered sums can y cover?
                cov = 0
                for s in uncovered_sums:
                    z = s - y
                    if 0 <= z <= n and (z in A_greedy or z == y):
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_y = y
            if best_y is not None and best_cov > 0:
                A_greedy.add(best_y)
                x_g = init_x(A_greedy)
                counts_g = compute_full_counts_fft(x_g)
            else:
                break
        inits.append(A_greedy)
        
        return inits

    # --- Main ---
    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    
    # Generate initial sets
    inits = generate_initial_sets()
    
    # Evaluate all inits and sort
    scored_inits = []
    for A_init in inits:
        if time.time() > deadline - 5.0:
            break
        x_init = init_x(A_init)
        counts_init = compute_full_counts_fft(x_init)
        score_init = get_score(counts_init)
        scored_inits.append((score_init, A_init))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Run fast local search on top initial sets
    for score_init, A_init in scored_inits[:8]:
        if time.time() > deadline - 10.0:
            break
        A_res, score_res = fast_local_search(A_init, max_steps=150000)
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
    
    # If no valid solution found, fall back
    if best_A is None:
        # Return a simple valid set
        return list(range(0, n + 1))
    
    # Final check: ensure all target sums are covered
    x_final = init_x(best_A)
    counts_final = compute_full_counts_fft(x_final)
    if np.any(counts_final[target_lo:target_hi + 1] == 0):
        # Try to fix by adding elements
        for y in range(n + 1):
            if y not in best_A:
                x_final[y] = 1
                counts_new = compute_full_counts_fft(x_final)
                if not np.any(counts_new[target_lo:target_hi + 1] == 0):
                    best_A.add(y)
                    break
                else:
                    x_final[y] = 0
        # If still not covered, return full set
        if np.any(compute_full_counts_fft(init_x(best_A))[target_lo:target_hi + 1] == 0):
            return list(range(0, n + 1))
    
    return list(best_A)
