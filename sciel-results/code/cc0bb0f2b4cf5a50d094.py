import time
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but we should be safe)
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N_ELEMENTS = 129  # 0 to 128

    # Precompute pair contributions for speed
    # pair_contrib[i][j] is a bitmask or list of sums? No, let's precompute the multiplicity impact.
    # Actually, for ILP we need the matrix A where A[t][i] = 1 if element i contributes to sum t.
    # Wait, the multiplicity of sum s is the number of pairs (a,b) with a+b=s.
    # This is a quadratic objective/constraint if we view it directly.
    # However, the strategy suggests: "Model the problem as a minimum maximum-flow network design... use max-flow/min-cut... then solve resulting set cover via ILP".
    # This is a complex heuristic. Let's stick to the "smooth surrogate" or "auxiliary variable" idea which is more tractable for a quick improvement.
    
    # Let's try a different approach:
    # 1. Fast local search with a better initialization.
    # 2. Use a "soft" peak minimization: minimize sum(exp(alpha * M[t])) or similar?
    # 3. Or simply improve the local search by looking at 2-swap moves and using a more directed search for gaps.
    
    # The current score is 8.25 (seed/value). Seed value is 33. So current value is 33/8.25 = 4.
    # Wait, "search score is seed/value". If seed value is 33, and score is 8.25, then 33/value = 8.25 => value = 4.
    # So the current program finds a set with peak multiplicity 4.
    # Can we get 3? Or 2?
    # If we find value=3, score = 33/3 = 11.
    # If we find value=2, score = 33/2 = 16.5.
    # If we find value=1, score = 33.
    
    # Let's analyze the lower bound.
    # Number of pairs in A of size k is k(k+1)/2.
    # Total representations available = k(k+1)/2.
    # We need to cover 129 sums.
    # If peak multiplicity is P, then total representations >= 129 * 1 = 129 (for coverage) and also total reps <= 129 * P.
    # So k(k+1)/2 >= 129.
    # k=16: 136 reps. 136/129 ~ 1.05. So average multiplicity is just above 1.
    # So P=1 is theoretically possible if we can distribute 136 reps over 129 sums such that no sum has more than 1.
    # This implies exactly 129 sums have 1 rep, and 7 sums have 0? No, every sum MUST be covered.
    # So every sum has >= 1 rep.
    # Total reps = 136. Sums = 129.
    # Sum of multiplicities = 136.
    # If max multiplicity is 1, then all 129 sums must have multiplicity 1, and total reps must be 129.
    # But k=16 gives 136 reps. So 7 reps are "extra".
    # This means at least 7 sums must have multiplicity >= 2.
    # So P=1 is IMPOSSIBLE for k=16.
    # What about k=15? 15*16/2 = 120 reps. Not enough to cover 129 sums (need at least 129 reps).
    # So k must be at least 16.
    # For k=16, min possible peak is at least ceil(136/129) = 2?
    # No, min peak is determined by the distribution.
    # If we have 136 items into 129 bins, min max bin size is 2 (if 7 bins have 2, 122 bins have 1).
    # So P=2 is the theoretical lower bound for k=16.
    # Can we achieve P=2?
    # If yes, score = 33/2 = 16.5.
    # If we can achieve P=1, it's impossible.
    # So target is P=2.
    
    # Let's try to find a set of size 16 with peak multiplicity 2.
    
    def calc_mult(A_list):
        """A_list must be sorted. Returns multiplicity array for sums in [MIN_SUM, MAX_SUM]."""
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if s > MAX_SUM:
                    break
                if s >= MIN_SUM:
                    M[s - MIN_SUM] += 1
        return M

    def calc_score(A):
        A = sorted(A)
        M = calc_mult(A)
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)

    def local_search(A, deadline, max_iters=1000):
        A = sorted(list(A))
        M = calc_mult(A)
        current_score = calc_score(A)
        
        if current_score == (0, 1):
            return current_score, A, M
        
        improved = True
        iters = 0
        while improved and time.time() < deadline and iters < max_iters:
            improved = False
            iters += 1
            
            # Try 1-swap
            A_set = set(A)
            available = [x for x in range(129) if x not in A_set]
            
            # Shuffle available to avoid bias
            random.shuffle(available)
            
            best_neighbor_score = current_score
            best_neighbor_A = None
            best_neighbor_M = None
            
            # Limit swaps to speed up
            n_swaps = min(20, len(available))
            
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                a = A[i]
                for b in available[:n_swaps]:
                    new_A = A.copy()
                    new_A[i] = b
                    new_A.sort()
                    new_M = calc_mult(new_A)
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
                
                # If we hit the theoretical lower bound for this size, stop
                if current_score == (0, 2) and len(A) == 16:
                     # We might still try to find (0,1) but it's impossible for k=16.
                     # But let's not break yet, maybe a different k is better?
                     # Actually, if we find (0,2), we are at the lower bound for k=16.
                     # Let's try to see if k=17 allows (0,2) with more flexibility?
                     # Or k=16 with (0,2) is the best possible.
                     pass
                    
        return current_score, A, M

    def targeted_removal(A, M, deadline):
        """Try to remove elements if it doesn't increase peak or create gaps."""
        A = list(A)
        current_max = int(np.max(M))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(A)):
                if time.time() > deadline:
                    break
                new_A = A[:idx] + A[idx+1:]
                if not new_A:
                    continue
                new_M = calc_mult(new_A)
                if np.all(new_M >= 1):
                    new_max = int(np.max(new_M))
                    if new_max <= current_max:
                        A = new_A
                        M = new_M
                        current_max = new_max
                        improved = True
                        break
        return A, M

    best_score = (float('inf'), float('inf'))
    best_A = None
    
    # Try sizes 16 to 20
    # k=16 is the minimum size for coverage.
    # If we find (0,2) for k=16, that's score 16.5.
    # If we find (0,3) for k=16, that's score 11.
    # Current is 4 (score 8.25).
    
    # Let's prioritize finding (0,2).
    
    for k in range(16, 22):
        if time.time() > deadline:
            break
            
        # Number of restarts
        # For k=16, we need to be very careful.
        n_restarts = 50 if k == 16 else 20
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            A = sorted(random.sample(range(129), k))
            
            # Local search
            current_score, A, M = local_search(A, deadline, max_iters=200)
            
            # Targeted removal to see if we can shrink while maintaining score
            # But we are iterating k, so maybe just keep it.
            
            if current_score < best_score:
                best_score = current_score
                best_A = list(A)
                
            # Early exit if we hit the theoretical minimum
            if best_score == (0, 2):
                # Can we do better? No, P=1 is impossible for k=16.
                # Is it possible for k>16 to have P=1?
                # k=17: 153 reps. 153/129 = 1.18. Still P>=2.
                # k=18: 171 reps. 171/129 = 1.32. P>=2.
                # ...
                # So P=1 is never possible for k such that k(k+1)/2 > 129.
                # Wait, if k is large, we can remove elements.
                # But we start with k elements.
                # The minimum peak is determined by the total number of pairs.
                # If we have a set A of size k, the number of pairs is fixed.
                # So if we find (0,2), we are at the global minimum for that k.
                # And since P=1 is impossible for any k where pairs > 129, and we need pairs >= 129 for coverage,
                # P=1 is impossible globally?
                # Let's check k=15. Pairs=120. Cannot cover 129 sums.
                # So P=1 is indeed impossible.
                # Thus (0,2) is the global optimum.
                return sorted(best_A)
                
    # If we didn't find (0,2), return the best we found
    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    # Final polish
    if time.time() < deadline - 5:
         # Try to improve best_A further
         final_score, final_A, _ = local_search(best_A, deadline, max_iters=500)
         if final_score < best_score:
             best_score = final_score
             best_A = final_A
             
    return sorted(best_A)

# Run it once to test? No, just define.
# The evaluator calls construct().

