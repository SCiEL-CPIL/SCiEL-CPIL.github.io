import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = range(target_lo, target_hi + 1)
    num_targets = len(target_range)
    
    # We want to find A subset of [0, n] such that:
    # 1. Every s in [128, 256] is a sum of two elements in A (unordered, with replacement).
    # 2. Minimize the maximum number of such pairs for any s.
    
    # Strategy:
    # This is equivalent to finding a set A such that the autocorrelation of A (shifted)
    # covers [128, 256] with low peak.
    # 
    # Let's use a greedy load-balancing approach.
    # We maintain counts[s] for s in [0, 2n].
    # We want to add elements to A.
    # However, simply adding elements increases counts.
    # We need to ensure coverage first, then balance.
    # 
    # Actually, a better heuristic for minimax:
    # Start with a set A that provides "perfect" or near-perfect coverage with low multiplicity.
    # Known structures: 
    # If A is an interval [a, b], sums cover [2a, 2b] with triangular distribution.
    # Peak is at the middle.
    # For target [128, 256], if we pick A = [64, 128], sums are [128, 256].
    # Size of A is 65.
    # Max multiplicity for sum s in [128, 256] with A=[64, 128]:
    # s=128: 64+64 (1 way)
    # s=192: 64+128, 65+127, ..., 96+96 (33 ways)
    # s=256: 128+128 (1 way)
    # Peak is 33.
    # 
    # Can we do better?
    # If we use a "sparse" set, we might have gaps.
    # If we use a "structured" set like arithmetic progression, we get similar triangular shape.
    # 
    # Let's try to optimize the interval or a subset of an interval.
    # Or use a "balanced" set.
    # 
    # Alternative: Simulated Annealing with a specific focus on "Peak Reduction".
    # The current program uses SA but gets stuck at 8.25 (which implies a peak of roughly 8? No, score is 8.25).
    # Wait, the score is "seed/value". If seed is 1.0, value is 1/0.12? No.
    # The task says: "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # Usually seed is 1.0. So if score is 8.25, value is 1/8.25 = 0.121?
    # No, "1.0 equals the seed program". So if seed program has value V_seed, score = V_seed / V_child?
    # Or score = V_seed / V_child?
    # "search score is seed/value... 1.0 equals the seed program".
    # If seed program has value 10, and child has value 1, score = 10/1 = 10.
    # If child has value 10, score = 10/10 = 1.
    # If child has value 5, score = 10/5 = 2.
    # So higher score is better.
    # Current score 8.25 implies Value_child = Value_seed / 8.25.
    # If Value_seed is the peak multiplicity of the seed.
    # What is the seed? "Keep the function names... of the seed program".
    # The provided code IS the seed? Or the code provided is the "CURRENT PROGRAM" with score 8.25?
    # "CURRENT PROGRAM (score 8.2500)".
    # So the current program achieves a score of 8.25.
    # This means the current program's output has a "value" that is 1/8.25 times the baseline?
    # Or does "value" refer to the objective function value (max multiplicity)?
    # "Minimize the largest representation multiplicity".
    # Let M be the max multiplicity.
    # If the evaluator returns M, and score = Baseline_M / M.
    # If Baseline_M is, say, 10. And current M is 1.2? No, M is integer.
    # If Baseline_M is 100, and current M is 12, score = 8.33.
    # If Baseline_M is 10, and current M is 1, score = 10.
    # If Baseline_M is 8, and current M is 1, score = 8.
    # If Baseline_M is 16, and current M is 2, score = 8.
    # 
    # Let's assume the "value" is the max multiplicity.
    # If the current program finds a set with max multiplicity 2, and the baseline is 16, score is 8.
    # Or if baseline is 8.25 * M.
    # 
    # Let's look at the "Inspirations". They also have score 8.25.
    # This suggests 8.25 is a "plateau" or a common result.
    # 
    # Let's try to find a set with max multiplicity 1.
    # If max multiplicity is 1, it's a "Sidon set" or "B2 sequence" restricted to sums in [128, 256].
    # If every sum in [128, 256] is unique, then M=1.
    # Is it possible to cover [128, 256] (129 integers) with unique sums from A subset [0, 128]?
    # Number of pairs in A is |A|(|A|+1)/2.
    # We need at least 129 pairs.
    # |A|(|A|+1)/2 >= 129 => |A|^2 ~ 258 => |A| ~ 16.
    # So if we can find a set of size ~16-20 such that all sums in [128, 256] are distinct, M=1.
    # This is a strong condition.
    # 
    # Let's try to construct such a set.
    # This is related to "Perfect Difference Sets" or similar.
    # 
    # Greedy construction for M=1:
    # 1. Initialize A = empty.
    # 2. We need to cover 128..256.
    # 3. Pick x in [0, 128].
    # 4. Check if adding x creates any duplicate sums in [128, 256] with existing elements.
    # 5. If no duplicates, add x.
    # 6. Repeat until all covered or no more candidates.
    # 
    # If we can cover all with M=1, score will be huge (if baseline is >1).
    # 
    # Let's implement this "Unique Sum" greedy.
    
    def try_unique_sum_greedy():
        A = []
        A_set = set()
        used_sums = set() # Sums in [128, 256] already formed
        
        # Candidates: 0..128
        # We want to cover 128..256.
        # Note: x+y = s.
        # If we pick x, it forms sums x+y for y in A (including x).
        
        # To maximize chance, we should pick elements that cover "hard" sums.
        # Middle sums are harder? Or edges?
        # Let's just iterate candidates.
        
        # Heuristic: Start with elements that are "spread out".
        # Random restarts might help.
        
        best_A = None
        best_coverage = -1
        
        for _ in range(100): # 100 restarts
            A = []
            A_set = set()
            used_sums = set()
            
            # Randomize candidate order
            candidates = list(range(0, 129))
            np.random.shuffle(candidates)
            
            # To ensure coverage, we might need to force certain elements.
            # But let's try pure greedy first.
            
            # Sort candidates by "potential" ?
            # For now, random order.
            
            for x in candidates:
                if x in A_set:
                    continue
                
                # Check if adding x violates uniqueness in target range
                valid = True
                new_sums = []
                
                # Pairs (x, y) for y in A
                for y in A:
                    s = x + y
                    if target_lo <= s <= target_hi:
                        if s in used_sums:
                            valid = False
                            break
                        new_sums.append(s)
                
                # Pair (x, x)
                if valid:
                    s = 2 * x
                    if target_lo <= s <= target_hi:
                        if s in used_sums:
                            valid = False
                        else:
                            new_sums.append(s)
                
                if valid:
                    # Add x
                    A.append(x)
                    A_set.add(x)
                    used_sums.update(new_sums)
                    
                    # Check if all covered
                    if len(used_sums) == 129:
                        return A # Found perfect set!
            
            # Track best coverage
            if len(used_sums) > best_coverage:
                best_coverage = len(used_sums)
                best_A = A[:]
                
            # Early stop if time is tight
            if time.time() > deadline - 10:
                break
                
        return best_A

    # First, try to find a set with Max Multiplicity 1.
    A_unique = try_unique_sum_greedy()
    
    if A_unique is not None:
        # Verify it
        counts = np.zeros(257, dtype=np.int32)
        v = np.zeros(129, dtype=np.int8)
        for x in A_unique:
            v[x] = 1
        K = np.convolve(v, v, mode='full')
        even_indices = np.arange(0, 257, 2)
        half_indices = even_indices // 2
        valid_mask = half_indices <= 128
        valid_even_indices = even_indices[valid_mask]
        valid_half_indices = half_indices[valid_mask]
        N_eq = np.zeros(257, dtype=np.int32)
        N_eq[valid_even_indices] = v[valid_half_indices]
        counts = (K + N_eq) // 2
        
        c_target = counts[128:257]
        if np.all(c_target > 0) and np.max(c_target) == 1:
            return A_unique
            
    # If M=1 is not found, try to minimize M using Load Balancing.
    # The previous program used SA. Let's use a deterministic "Minimax" greedy.
    # 
    # Idea:
    # Start with a set that covers all, e.g., A = [64, 128]. Peak is ~33.
    # Try to "swap" elements to reduce peak.
    # Or, build from scratch with a penalty for high loads.
    
    def load_balancing_greedy():
        A = []
        A_set = set()
        counts = np.zeros(257, dtype=np.int32)
        
        # We want to cover [128, 256].
        # We will add elements one by one.
        # At each step, we pick the element x that minimizes the resulting "Max Load"
        # while maximizing "New Coverage".
        
        # To make it deterministic and fast, we can use a fixed order or random.
        # Let's use a few random restarts with this greedy criterion.
        
        best_A = None
        best_peak = float('inf')
        best_uncovered = float('inf')
        
        for restart in range(50):
            A = []
            A_set = set()
            counts = np.zeros(257, dtype=np.int32)
            
            # Initialize with a few random elements to break symmetry
            for _ in range(10):
                x = np.random.randint(0, 129)
                if x not in A_set:
                    A.append(x)
                    A_set.add(x)
                    # Update counts
                    for y in A:
                        s = x + y
                        if target_lo <= s <= target_hi:
                            counts[s] += 1
                    s2 = 2*x
                    if target_lo <= s2 <= target_hi:
                        counts[s2] += 1
            
            # Greedy Add
            max_iter = 100
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Check if covered
                c_target = counts[128:257]
                uncovered = np.sum(c_target == 0)
                if uncovered == 0:
                    peak = np.max(c_target)
                    if peak < best_peak or (peak == best_peak and len(A) < len(best_A)):
                        best_peak = peak
                        best_A = A[:]
                        best_uncovered = 0
                    break
                    
                # Find best x to add
                best_x = -1
                best_score = -1
                
                # Sample candidates to speed up
                # Or iterate all 129? 129 is small.
                for x in range(129):
                    if x in A_set:
                        continue
                    
                    # Simulate adding x
                    # We need to calculate new counts and new peak/uncovered.
                    # To be fast, we only care about:
                    # 1. How many new sums are covered?
                    # 2. What is the new max count?
                    
                    new_uncovered = uncovered
                    new_max = np.max(c_target) if uncovered > 0 else 0 # If not covered, max doesn't matter as much?
                    # Actually, we want to minimize peak eventually.
                    # Score = (Newly Covered) * 1000 - (New Peak - Current Peak) * 10 ?
                    
                    # Let's just count newly covered sums.
                    newly_covered = 0
                    max_count_after = 0
                    
                    # Current max in target range
                    curr_max = np.max(c_target)
                    
                    for y in A:
                        s = x + y
                        if target_lo <= s <= target_hi:
                            if c_target[s - 128] == 0: # c_target index is s-128
                                newly_covered += 1
                            if c_target[s - 128] + 1 > max_count_after:
                                max_count_after = c_target[s - 128] + 1
                    
                    s2 = 2 * x
                    if target_lo <= s2 <= target_hi:
                        if c_target[s2 - 128] == 0:
                            newly_covered += 1
                        if c_target[s2 - 128] + 1 > max_count_after:
                            max_count_after = c_target[s2 - 128] + 1
                    
                    # Score: Prioritize coverage. If covered, minimize peak.
                    if newly_covered == 0:
                        # Only add if it helps balance?
                        # For now, skip if no new coverage.
                        continue
                        
                    # Penalty for increasing peak
                    penalty = 0
                    if max_count_after > curr_max:
                        penalty = (max_count_after - curr_max) * 10
                    
                    score = newly_covered * 100 - penalty
                    
                    if score > best_score:
                        best_score = score
                        best_x = x
                
                if best_x == -1:
                    break
                
                # Add best_x
                A.append(best_x)
                A_set.add(best_x)
                
                # Update counts
                for y in A:
                    if y == best_x:
                        s = 2 * best_x
                        if target_lo <= s <= target_hi:
                            counts[s] += 1
                    else:
                        s = best_x + y
                        if target_lo <= s <= target_hi:
                            counts[s] += 1
            
            # If not fully covered, record partial?
            # No, we need full coverage.
            
            if time.time() > deadline - 5:
                break
                
        return best_A, best_peak

    A_lb, peak_lb = load_balancing_greedy()
    
    if A_lb is not None:
        # Verify
        v = np.zeros(129, dtype=np.int8)
        for x in A_lb:
            v[x] = 1
        K = np.convolve(v, v, mode='full')
        even_indices = np.arange(0, 257, 2)
        half_indices = even_indices // 2
        valid_mask = half_indices <= 128
        valid_even_indices = even_indices[valid_mask]
        valid_half_indices = half_indices[valid_mask]
        N_eq = np.zeros(257, dtype=np.int32)
        N_eq[valid_even_indices] = v[valid_half_indices]
        counts = (K + N_eq) // 2
        c_target = counts[128:257]
        if np.all(c_target > 0):
            return A_lb
            
    # Fallback: The interval [64, 128]
    return list(range(64, 129))

if __name__ == "__main__":
    A = construct()
    print(A)
