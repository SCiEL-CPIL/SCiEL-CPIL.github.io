import time
import random
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
    
    # Precompute sum indices for each potential element x in [0, 128]
    # sums_for_x[x] is a list of target sums s in [128, 256] such that s-x is in [0, 128]
    # Actually, we need to know which pairs (x, y) contribute to which sums.
    # For a given x, the partners y are in [0, 128].
    # The sum is s = x + y.
    # We care about s in [128, 256].
    
    # Let's precompute for each x, the list of (y, s) pairs that are valid targets.
    # But y varies.
    # Better: For each x, precompute the list of sums s that x contributes to IF y is chosen.
    # Wait, the count of s depends on the number of pairs (x, y) in A such that x+y=s.
    
    # Let's use a different formulation for the heuristic:
    # We want to select A subset of [0, 128].
    # Let c[s] be the number of pairs {x, y} in A such that x+y=s.
    # We want to minimize max(c[s]) for s in [128, 256], subject to c[s] >= 1 for all s.
    
    # Heuristic:
    # 1. Start with A = empty.
    # 2. We must cover all sums.
    # 3. Instead of random, we can try to construct A such that the "load" on each sum is balanced.
    
    # A deterministic approach:
    # We can view this as a set cover problem with a balancing objective.
    # Greedy Set Cover:
    # At each step, pick the element x in [0, 128] \ A that covers the most "uncovered" sums,
    # with a tie-breaker that minimizes the increase in the current maximum count.
    
    # To speed up, we can precompute for each x, the set of sums it can form with ANY y in [0, 128].
    # But the contribution of x to c[s] depends on how many y in A satisfy x+y=s.
    
    # Let's try a different angle:
    # Symmetry. If A is a solution, then 128-A is a solution?
    # If x, y in A, then x+y = s.
    # Let x' = 128-x, y' = 128-y.
    # x'+y' = 256 - (x+y) = 256 - s.
    # So the distribution of counts is symmetric around 128.
    # c[s] = c[256-s].
    
    # Let's use a simple greedy algorithm that iteratively adds elements.
    
    def get_initial_A():
        # Start with a sparse set that ensures coverage potential.
        # To cover [128, 256], we need elements spread out.
        # If A contains 0 and 128, we cover 128 and 256.
        # If A contains 64, we cover 128 (64+64), 192 (64+128), etc.
        
        # A good starting point might be multiples of some number?
        # Or just a random set that is guaranteed to cover?
        # A set of size 129 (all elements) covers everything with high counts.
        # We want a smaller set.
        
        # Let's start with a random set of size 40 and refine.
        # Or better, a structured set.
        # Try: A = {0, 1, ..., 32} U {96, 97, ..., 128}?
        # Sums from first part: [0, 64]. Sums from second: [192, 256].
        # Cross sums: [96, 160].
        # Missing: [65, 95] and [161, 191].
        # So we need to fill the middle.
        
        # Let's try A = {0, 32, 64, 96, 128} plus some noise?
        # Sums: 0, 32, 64, 96, 128, 160, 192, 224, 256.
        # Gaps are large.
        
        # Let's use a greedy coverage builder.
        A = set()
        counts = np.zeros(257, dtype=np.int32)
        
        # We will add elements one by one.
        # Score for adding x:
        # 1. Number of currently zero sums that become > 0.
        # 2. Penalty for increasing the max count.
        
        # To make it fast, we can precompute for each x, the sums it forms with current A.
        
        # Initialize with a few random elements to break symmetry
        for _ in range(10):
            x = random.randint(0, n)
            A.add(x)
            
        # Recalculate counts for initial A
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        
        # Greedy addition
        for _ in range(100): # Max 100 additions
            if time.time() > deadline - 10:
                break
                
            # Check if all covered
            c_target = counts[target_lo:target_hi+1]
            if np.all(c_target > 0):
                break
                
            best_x = -1
            best_score = -1
            
            # Try all x not in A
            for x in range(n + 1):
                if x in A:
                    continue
                    
                # Calculate potential counts if x is added
                # New sums formed: x + y for y in A, and x + x.
                # We need to count how many of these are currently 0.
                
                newly_covered = 0
                max_increase = 0
                current_max = np.max(c_target) if len(c_target) > 0 else 0
                
                # Iterate y in A
                for y in A:
                    s = x + y
                    if target_lo <= s <= target_hi:
                        if counts[s] == 0:
                            newly_covered += 1
                        # Track max count after addition
                        # counts[s] will become counts[s] + 1
                        if counts[s] + 1 > current_max:
                            if counts[s] + 1 > max_increase:
                                max_increase = counts[s] + 1
                
                # Self sum
                s = 2 * x
                if target_lo <= s <= target_hi:
                    if counts[s] == 0:
                        newly_covered += 1
                    if counts[s] + 1 > current_max:
                        if counts[s] + 1 > max_increase:
                            max_increase = counts[s] + 1
                
                # Score: Prioritize coverage, then minimize peak
                # If newly_covered is 0, we might still add to balance?
                # For now, only add if it covers something.
                if newly_covered == 0:
                    continue
                    
                # Penalty if peak increases significantly
                # Ideal peak is around total_pairs / num_targets.
                # Let's just minimize the new peak.
                score = newly_covered * 1000 - max_increase * 10
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x == -1:
                break
                
            # Add best_x
            A.add(best_x)
            A_list = list(A)
            
            # Update counts
            # We can just recalculate from scratch if A is small, or update incrementally.
            # Incremental:
            for y in A:
                if y == best_x:
                    continue # Handled below or double counted?
                s = best_x + y
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s = 2 * best_x
            if target_lo <= s <= target_hi:
                counts[s] += 1
        
        return A, counts

    # Run the greedy builder multiple times with different random seeds
    best_A = None
    best_peak = float('inf')
    
    for i in range(20):
        if time.time() > deadline - 5:
            break
            
        random.seed(i)
        A, counts = get_initial_A()
        
        # Check validity
        c_target = counts[target_lo:target_hi+1]
        if not np.all(c_target > 0):
            continue
            
        peak = np.max(c_target)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A
            
        # Early stop if we get a very low peak
        # Theoretical lower bound:
        # |A| = k. Number of pairs is k(k+1)/2.
        # Number of targets is 129.
        # Average count = k(k+1)/2 / 129.
        # If k=20, pairs=210, avg=1.6. Peak must be at least 2.
        # If k=30, pairs=465, avg=3.6. Peak at least 4.
        # If k=40, pairs=820, avg=6.3. Peak at least 7.
        
        if best_peak <= 3:
            break

    if best_A is None:
        # Fallback: Return a set that definitely covers, even if peak is high.
        # e.g., all even numbers?
        # 0, 2, ..., 128.
        # Sums are even. 128, 130, ..., 256.
        # Odd sums 129, 131, ... 255 are NOT covered.
        # So we need both parities.
        # Let's return range(0, 129) as a last resort.
        best_A = set(range(129))
        
    return list(best_A)

if __name__ == "__main__":
    A = construct()
    print(A)
