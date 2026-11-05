"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    
    # Theoretical lower bound for peak multiplicity:
    # Total pairs = sum over all sums of multiplicity.
    # If A has size m, max pairs ~ m^2/2.
    # Target range size = 129.
    # If we pick m=40, pairs ~ 800. 800/129 ~ 6.
    # Let's try to find a solution with low peak.
    
    best_A = None
    best_peak = float('inf')
    
    # We will try to find a solution with peak multiplicity <= K
    # We start with a reasonable K and decrease if we find a solution quickly?
    # Or start low and increase?
    # Greedy set cover usually works best if we don't constrain too tight initially, 
    # but we want to MINIMIZE peak. 
    # Let's try a range of K values.
    
    # Estimate lower bound:
    # Each element a in A contributes to sums a+x and a+y.
    # To cover [128, 256], we need sufficient density.
    
    # Let's attempt a greedy construction with a hard constraint on peak multiplicity.
    # We will try K from a low estimate up to a safe high value.
    # If we find a valid A for K, that's a good candidate.
    # We want the minimum K.
    
    # Lower bound estimation:
    # If A is a subset of [0, 128], let's say size m.
    # Number of pairs is m(m+1)/2.
    # These pairs must cover 129 sums.
    # If peak is K, then total capacity is 129*K.
    # So m(m+1)/2 <= 129*K is not a constraint, it's the other way:
    # We need enough pairs to cover. 129*K >= number of required distinct sums? No.
    # We need to cover every sum. 
    # A sum s can be represented by at most K pairs.
    # Total pairs generated = sum(count[s] for s in target).
    # This sum must be at least 129 (to cover each at least once).
    # Also, total pairs <= m(m+1)/2.
    # So m(m+1)/2 >= 129. m ~ 16.
    # But we also need the pairs to land in the target interval.
    # Sums < 128 or > 256 are "wasted" for coverage but count towards multiplicity if we are not careful?
    # No, multiplicity is only counted for sums in [128, 256].
    # Pairs summing outside [128, 256] do not contribute to the cost (peak multiplicity in target interval).
    # However, they consume elements of A.
    
    # Strategy:
    # 1. Pick a target peak K.
    # 2. Greedily build A.
    #    - Maintain count[s] for s in [128, 256].
    #    - Maintain list of uncovered sums.
    #    - At each step, evaluate all candidate x in [0, 128] \ A.
    #    - For each candidate x, calculate new_counts if x is added.
    #    - Check if any new_counts[s] > K for s in target. If so, reject x.
    #    - Among valid x, pick the one that covers the most uncovered sums (or weighted by scarcity).
    #    - If no valid x, fail.
    #    - If all sums covered, success.
    
    # We need to search for the minimal K.
    # Let's start with K=1? Unlikely. K=2? Maybe.
    # Let's try K values from 1 to 10.
    # If K is too small, greedy might get stuck.
    # We can add randomization to break ties or restart.
    
    def try_greedy(K, seed):
        random.seed(seed)
        A = []
        counts = np.zeros(max_sum + 1, dtype=int)
        uncovered = set(target_sums)
        
        # Precompute potential contributions? 
        # Since A grows, we just check candidates.
        
        while uncovered:
            if time.time() > deadline - 1.0:
                return None, float('inf')
            
            best_candidate = None
            best_score = -1
            best_new_counts = None
            best_new_uncovered = None
            
            candidates = [x for x in range(n + 1) if x not in A]
            # To speed up, we might restrict candidates to those that can actually help.
            # A candidate x helps if there exists y in A or y=x such that x+y in uncovered.
            # Or simply x+x in uncovered.
            
            # Optimization: Only consider x such that x+x >= min_sum or x+min(A) >= min_sum?
            # Actually, if x is very small, x+y might be < 128 for all y in A.
            # Let's just filter candidates that cover at least one uncovered sum.
            
            valid_candidates = []
            for x in candidates:
                # Check if x can cover any uncovered sum
                # Sums involving x: x+x, x+y for y in A.
                covers = False
                for s in uncovered:
                    if s == 2*x:
                        covers = True
                        break
                    y = s - x
                    if y in A:
                        covers = True
                        break
                    # Note: unordered pairs, so if y in A, (x,y) is a pair.
                    # We don't need to check y > x specifically for coverage existence, 
                    # just that the pair exists.
                
                if covers:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None, float('inf')
            
            # Score each valid candidate
            for x in valid_candidates:
                # Calculate impact
                # New pairs: (x, x) and (x, y) for y in A.
                # Note: if y=x, it's (x,x). If y!=x, it's (x,y).
                # Since we add x to A, the new pairs are:
                # 1. (x, x)
                # 2. (x, y) for each y in A where y != x (which is all y in A since x not in A yet)
                
                new_counts = counts.copy()
                
                # Add (x,x)
                s_xx = 2*x
                if min_sum <= s_xx <= max_sum:
                    new_counts[s_xx] += 1
                    if new_counts[s_xx] > K:
                        continue # Invalid
                
                # Add (x, y) for y in A
                for y in A:
                    s_xy = x + y
                    if min_sum <= s_xy <= max_sum:
                        new_counts[s_xy] += 1
                        if new_counts[s_xy] > K:
                            break # Invalid, restart loop for this x
                else:
                    # If we didn't break, it's valid so far
                    # Calculate score: number of newly covered sums
                    # A sum is newly covered if it was in uncovered and new_counts[s] >= 1 
                    # (which it is if we just added to it, or if it was already covered? No, we only care about UNCOVERED becoming covered)
                    
                    # Actually, simpler: count how many sums in `uncovered` have new_counts[s] > 0.
                    # But we must ensure we don't count sums that were already covered?
                    # `uncovered` contains only sums with count 0.
                    # So any s in `uncovered` with new_counts[s] > 0 is newly covered.
                    
                    new_cov_count = 0
                    for s in uncovered:
                        if new_counts[s] > 0:
                            new_cov_count += 1
                    
                    if new_cov_count > best_score:
                        best_score = new_cov_count
                        best_candidate = x
                        best_new_counts = new_counts
                        # We don't strictly need best_new_uncovered if we recalculate, 
                        # but we need to update it.
                        # For now, just store candidate. We will re-evaluate uncovered after adding.
            
            if best_candidate is None:
                return None, float('inf')
            
            A.append(best_candidate)
            counts = best_new_counts
            
            # Update uncovered
            new_uncovered = set()
            for s in uncovered:
                if counts[s] == 0:
                    new_uncovered.add(s)
            uncovered = new_uncovered
            
            if not uncovered:
                return A, K
        
        return None, float('inf')
    
    # Search for minimal K
    # Let's try K from 1 upwards.
    # We can run multiple seeds for each K to handle greedy traps.
    
    # Heuristic lower bound for K:
    # If we pick m elements, we have ~m^2/2 pairs.
    # To cover 129 sums with peak K, we need 129*K >= 129 (trivial).
    # But we also need the geometry to work.
    # Let's just iterate K.
    
    found = False
    # Try K=1, 2, 3... up to a reasonable limit (e.g., 10)
    # If we don't find a solution for small K, we might need larger K or better search.
    # Given the 90s limit, we should be careful.
    
    # Let's try K=1 first? 
    # If K=1, it's a Sidon set-like constraint for the sums in the interval.
    # This is very restrictive.
    
    # Let's try a range of K and seeds.
    # We will prioritize smaller K.
    
    # To save time, we can pre-check if a solution is possible.
    # But let's just run the greedy.
    
    # We will run for K in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # For each K, try a few seeds.
    
    # Time budget per K?
    # Total time 85s.
    # Let's allocate time based on K.
    
    # Actually, if K is too small, it fails fast. If K is large, it succeeds fast.
    # So iterating K from low to high is efficient.
    
    for K in range(1, 11):
        if time.time() > deadline - 5.0:
            break
        
        # Try a few seeds for this K
        # If K is small, we might need more restarts.
        num_seeds = 5 if K < 5 else 2
        
        for seed in range(num_seeds):
            if time.time() > deadline - 1.0:
                break
            
            A, peak = try_greedy(K, seed)
            if A is not None:
                # Verify the solution
                # Check coverage
                counts = np.zeros(max_sum + 1, dtype=int)
                A_sorted = sorted(A)
                for i in range(len(A_sorted)):
                    ai = A_sorted[i]
                    if min_sum <= 2*ai <= max_sum:
                        counts[2*ai] += 1
                    for j in range(i+1, len(A_sorted)):
                        s = ai + A_sorted[j]
                        if min_sum <= s <= max_sum:
                            counts[s] += 1
                
                is_valid = True
                max_mult = 0
                for s in target_sums:
                    if counts[s] == 0:
                        is_valid = False
                        break
                    max_mult = max(max_mult, counts[s])
                
                if is_valid:
                    if max_mult < best_peak:
                        best_peak = max_mult
                        best_A = A
                    # Since we iterate K from low to high, the first valid A found for a K 
                    # might not have peak exactly K, but <= K.
                    # We want the minimal peak. 
                    # If we found a solution with peak P <= K, and we are trying K, 
                    # we should check if we can do better.
                    # But if we found a solution for K=3, it might have peak 2.
                    # We should record the actual peak.
                    # If we find a solution with peak P, we can stop searching for K > P?
                    # No, we want to minimize P.
                    # If we found P=3 with K=3, maybe K=2 works with a different A?
                    # So we should continue checking smaller K if we haven't tried them?
                    # We are iterating K from 1 up.
                    # So if we find a solution for K=3, we have already tried K=1 and K=2.
                    # If they failed, then 3 is the best we found so far.
                    # But wait, try_greedy(K) enforces peak <= K.
                    # So if try_greedy(2) failed, there is no solution with peak <= 2 found by this greedy.
                    # It doesn't prove non-existence, but it's a strong heuristic.
                    # So the first K that succeeds gives a solution with peak <= K.
                    # We record the actual peak.
                    # We can break out of K loop if we find a solution with peak == K?
                    # No, because a solution with peak K might exist, but maybe a solution with peak K-1 exists that greedy missed?
                    # Unlikely if greedy is decent.
                    # Let's just take the best found so far.
                    
                    # If we found a solution, we can return it if we are confident.
                    # But we want to minimize.
                    # Let's finish the loop to see if a smaller K works (which we already tried).
                    # Since we tried smaller K first, if we are at K=5 and found a solution, 
                    # we know K=1..4 failed for this greedy.
                    # So this is likely the minimal peak for this greedy strategy.
                    # We can break.
                    break
        
        if best_A is not None:
            # We found a solution for some K.
            # Since we iterate K ascending, this is the minimal K that worked.
            # The actual peak might be lower than K, but we can't do better than what greedy found for smaller K.
            break

    if best_A is None:
        # Fallback: return the seed solution if we failed to find a better one
        # Or a simple dense set.
        return list(range(64, 129))
    
    return best_A
# EVOLVE-BLOCK-END
