import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N = 129  # elements in [0, 128]
    
    def calc_mult_fft(A_set):
        """Use FFT convolution to compute multiplicity array for sums in [MIN_SUM, MAX_SUM]."""
        f = np.zeros(2 * N, dtype=np.float64)
        for a in A_set:
            f[a] = 1.0
        # Convolve f with itself
        conv = np.fft.irfft(np.fft.rfft(f) ** 2)
        # For unordered pairs (a <= b), we need:
        # - For a == b: count as 1 (not 2)
        # - For a != b: count as 1 (not 2, since we only count a <= b)
        # The convolution gives ordered pairs, so we need to adjust
        # conv[s] = number of ordered pairs (a,b) with a+b=s
        # For unordered with a<=b: if a!=b, each pair counted twice in conv, so divide by 2
        # if a==b, counted once in conv
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        for s in range(MIN_SUM, MAX_SUM + 1):
            ordered_count = int(round(conv[s]))
            # Count how many a in A_set such that 2*a == s
            diag = 1 if (s % 2 == 0 and s // 2 in A_set) else 0
            # Unordered count = (ordered_count - diag) / 2 + diag
            M[s - MIN_SUM] = (ordered_count - diag) // 2 + diag
        return M
    
    def calc_mult_fast(A_list):
        """A_list must be sorted. Returns multiplicity array for sums in [MIN_SUM, MAX_SUM]."""
        A_set = set(A_list)
        return calc_mult_fft(A_set)
    
    def calc_score(A):
        A = sorted(A)
        M = calc_mult_fast(A)
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)
    
    def get_elements_contributing_to_sum(A, target_sum):
        """Return list of indices in A that contribute to target_sum."""
        contributors = []
        A_set = set(A)
        for i, a in enumerate(A):
            b = target_sum - a
            if b in A_set and b >= a:
                contributors.append(i)
        return contributors
    
    def targeted_local_search(A, M, deadline, max_iters=500):
        """Local search that specifically targets high-multiplicity sums."""
        A = sorted(list(A))
        current_score = calc_score(A)
        
        if current_score == (0, 1):
            return current_score, A, M
        
        improved = True
        while improved and time.time() < deadline and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Find the highest multiplicity targets
            max_mult = int(np.max(M))
            if max_mult <= 1:
                break
            
            # Get targets with max multiplicity
            high_mult_targets = np.where(M == max_mult)[0] + MIN_SUM
            
            # For each high-mult target, try to reduce its multiplicity
            for target in high_mult_targets:
                if time.time() > deadline:
                    break
                
                contributors = get_elements_contributing_to_sum(A, target)
                if len(contributors) < 2:
                    continue
                
                # Try removing or swapping each contributor
                A_set = set(A)
                available = [x for x in range(N) if x not in A_set]
                
                for idx in contributors:
                    if time.time() > deadline:
                        break
                    a = A[idx]
                    
                    # Try removing this element
                    new_A = A[:idx] + A[idx+1:]
                    if new_A:
                        new_M = calc_mult_fast(new_A)
                        if np.all(new_M >= 1):
                            new_max = int(np.max(new_M))
                            new_score = (0, new_max)
                            if new_score < current_score:
                                current_score = new_score
                                A = new_A
                                M = new_M
                                improved = True
                                break
                    
                    # Try swapping with a random available element
                    if not improved and available:
                        n_swaps = min(8, len(available))
                        for b in random.sample(available, n_swaps):
                            new_A = A.copy()
                            new_A[idx] = b
                            new_A.sort()
                            new_M = calc_mult_fast(new_A)
                            gaps = int(np.sum(new_M == 0))
                            if gaps > 0:
                                continue
                            new_max = int(np.max(new_M))
                            new_score = (0, new_max)
                            if new_score < current_score:
                                current_score = new_score
                                A = new_A
                                M = new_M
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return current_score, A, M
    
    def greedy_removal(A, M, deadline, max_rounds=2000):
        """Try to remove elements while maintaining validity and not increasing peak."""
        A = list(A)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            if len(A) <= 1:
                break
            current_max = int(np.max(M))
            improved = False
            # Try removing each element, prefer those that reduce max
            for idx in range(len(A)):
                new_A = A[:idx] + A[idx+1:]
                if not new_A:
                    continue
                new_M = calc_mult_fast(new_A)
                if np.all(new_M >= 1):
                    new_max = int(np.max(new_M))
                    if new_max < current_max:
                        A = new_A
                        M = new_M
                        improved = True
                        break
                    elif new_max == current_max and random.random() < 0.1:
                        A = new_A
                        M = new_M
                        improved = True
                        break
            if not improved:
                break
        return A, M
    
    def local_search(A, deadline):
        A = sorted(list(A))
        M = calc_mult_fast(A)
        current_score = calc_score(A)
        
        if current_score == (0, 1):
            return current_score, A, M
        
        improved = True
        max_iters = 500
        
        while improved and time.time() < deadline and max_iters > 0:
            improved = False
            max_iters -= 1
            
            best_neighbor_score = current_score
            best_neighbor_A = None
            best_neighbor_M = None
            
            A_set = set(A)
            available = [x for x in range(N) if x not in A_set]
            
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                a = A[i]
                # Try random swaps
                n_candidates = min(10, len(available))
                if n_candidates == 0:
                    break
                candidates = random.sample(available, n_candidates)
                for b in candidates:
                    new_A = A.copy()
                    new_A[i] = b
                    new_A.sort()
                    new_M = calc_mult_fast(new_A)
                    gaps = int(np.sum(new_M == 0))
                    if gaps > 0:
                        new_score = (gaps, float('inf'))
                    else:
                        new_score = (0, int(np.max(new_M)))
                    if new_score < best_neighbor_score:
                        best_neighbor_score = new_score
                        best_neighbor_A = new_A
                        best_neighbor_M = new_M
                        
            if best_neighbor_score < current_score and best_neighbor_A is not None:
                current_score = best_neighbor_score
                A = best_neighbor_A
                M = best_neighbor_M
                improved = True
                if current_score == (0, 1):
                    break
                    
        return current_score, A, M
    
    best_score = (float('inf'), float('inf'))
    best_A = None
    best_M = None
    
    # Phase 1: Multi-restart local search with greedy removal
    for k in range(14, 30):
        if time.time() > deadline:
            break
        n_restarts = max(3, int(80 * (1 - (k - 14) / 16)))
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            A = sorted(random.sample(range(N), k))
            
            # Local search
            current_score, A, M = local_search(A, deadline)
            
            # Targeted local search to reduce peak multiplicity
            current_score, A, M = targeted_local_search(A, M, deadline, max_iters=300)
            
            # Greedy removal to shrink
            A, M = greedy_removal(A, M, deadline, max_rounds=500)
            current_score = calc_score(A)
            
            if current_score < best_score:
                best_score = current_score
                best_A = A
                best_M = M
                
            if best_score == (0, 1):
                return sorted(best_A)
    
    # Phase 2: Final targeted local search + greedy removal on best solution
    if best_A is not None:
        best_score, best_A, best_M = targeted_local_search(best_A, best_M, deadline, max_iters=500)
        best_A, best_M = greedy_removal(best_A, best_M, deadline, max_rounds=1000)
        final_score = calc_score(best_A)
        if final_score < best_score:
            best_score = final_score
    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return sorted(best_A)

