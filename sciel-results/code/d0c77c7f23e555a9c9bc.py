"""
Strategy: Simulated Annealing on Golomb Rulers.
Instead of searching for parameters of a geometric progression, we directly
optimize the set of integers U.
We aim to find a set U that is "Golomb-like" (distinct differences) to
maximize |U-U|, while tuning the maximum element (diameter) to balance
the numerator (ratio) and denominator (log diameter) of the score.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)

For a Golomb ruler of size n:
|U-U| = n(n-1) (all non-zero diffs distinct, plus 0)
|U+U| is typically around n(n+1)/2 if no sum collisions, but can be larger
or smaller depending on structure. Actually, for a Golomb ruler, sums
often have fewer collisions than arbitrary sets? No, sums of a Golomb ruler
can collide (e.g., 0+10 = 5+5).
However, maximizing |U-U| is the primary driver because it appears in the
numerator. Minimizing |U+U| is secondary but important.
Minimizing max(U) is important for the denominator.

We will use Simulated Annealing to:
1. Start with a random sparse set or a known small Golomb ruler.
2. Move elements to reduce sum collisions and maintain difference distinctness.
3. Adjust the overall scale (diameter) to find the sweet spot for the log term.
"""

import time
import math
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill
    
    # We need to find U.
    # Let's try sizes n from 10 to 50.
    # For each n, we try to find a "good" Golomb ruler and optimize its diameter.
    
    def compute_score(U):
        """
        Compute the score for a set U.
        """
        if not U:
            return 0.0
        U_set = set(U)
        if 0 not in U_set:
            return 0.0
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        # For very large n, O(n^2) might be slow, but we keep n < 50 in search.
        # n=50 -> 2500 pairs. Very fast.
        
        sumset = set()
        diffset = set()
        
        # Optimization: 
        # We can compute sums and diffs efficiently.
        # For n=50, 2500 iterations is trivial.
        
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
                
        # diffset includes negative values. |U-U| is the size of this set.
        # sumset includes non-negative values.
        
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    def score_params(U):
        return compute_score(U)

    best_score = 0.0
    best_U = [0, 1, 2]
    
    # We will search over different set sizes n.
    # The optimal n is likely small because log(2*maxU+1) grows with maxU,
    # and for large n, maxU must be large to keep differences distinct (Golomb ruler length ~ n^2).
    # Let's estimate:
    # If n=10, minimal Golomb ruler length is ~ 11 (e.g., 0,1,4,10,12,17,20,22,25,29 -> max 29).
    # max_U ~ n^2. log(n^2) ~ 2 log n.
    # |U-U| ~ n^2. |U+U| ~ n^2. Ratio ~ 1. Score ~ 1.
    # Wait, if Ratio > 1, Score > 1.
    # For a Golomb ruler, |U-U| = n(n-1)+1.
    # |U+U| is at most n(n+1)/2.
    # Ratio = (n^2-n+1) / (n^2/2) ~ 2.
    # log(2) / log(2 n^2) ~ 0.69 / (2 log n).
    # Score = 1 + 0.69 / (2 log n).
    # As n increases, Score approaches 1 from above.
    # So SMALL n is better?
    # Let's check n=3: U={0,1,3}. 
    # Sums: 0,1,2,3,4,6 -> {0,1,2,3,4,6} size 6.
    # Diffs: -3,-2,-1,0,1,2,3 -> size 7.
    # Ratio = 7/6 = 1.166.
    # max_U = 3. log(7) = 1.94.
    # Score = 1 + log(1.166)/log(7) = 1 + 0.153/1.94 = 1.079.
    
    # Let's check n=2: U={0, 10}.
    # Sums: 0, 10, 20 -> {0,10,20} size 3.
    # Diffs: -10, 0, 10 -> size 3.
    # Ratio = 1. Score = 1.
    
    # Let's check n=4: U={0, 1, 4, 10} (Golomb).
    # Sums: 
    # 0+0=0, 0+1=1, 0+4=4, 0+10=10
    # 1+0=1, 1+1=2, 1+4=5, 1+10=11
    # 4+0=4, 4+1=5, 4+4=8, 4+10=14
    # 10+0=10, 10+1=11, 10+4=14, 10+10=20
    # Sums set: {0, 1, 2, 4, 5, 8, 10, 11, 14, 20} size 10.
    # Diffs:
    # 0-0=0, 0-1=-1, 0-4=-4, 0-10=-10
    # 1-0=1, 1-1=0, 1-4=-3, 1-10=-9
    # 4-0=4, 4-1=3, 4-4=0, 4-10=-6
    # 10-0=10, 10-1=9, 10-4=6, 10-10=0
    # Diffs set: {-10, -9, -6, -4, -3, -1, 0, 1, 3, 4, 6, 9, 10} size 13.
    # Ratio = 13/10 = 1.3.
    # max_U = 10. log(21) = 3.04.
    # Score = 1 + log(1.3)/log(21) = 1 + 0.262/3.04 = 1.086.
    
    # n=5: U={0, 1, 4, 10, 12} (Golomb).
    # Let's calculate roughly.
    # |D| = 5*4+1 = 21.
    # |S|? 
    # If |S| is around 15, Ratio = 1.4.
    # max_U = 12. log(25) = 3.2.
    # Score = 1 + log(1.4)/3.2 = 1 + 0.336/3.2 = 1.105.
    
    # It seems n=5,6,7 might be optimal.
    # Let's search n from 3 to 20.
    # For each n, we use Simulated Annealing to find a set U of size n
    # that maximizes the score.
    
    # To speed up SA, we can fix the "shape" (relative positions) 
    # and only optimize the "scale" (multiplying by a factor)?
    # No, the integer nature matters for collisions.
    # But we can start with a known Golomb ruler and perturb it.
    
    # Known small Golomb rulers:
    # n=1: [0]
    # n=2: [0, 1]
    # n=3: [0, 1, 3]
    # n=4: [0, 1, 4, 6]
    # n=5: [0, 1, 4, 9, 11]
    # n=6: [0, 1, 4, 10, 12, 17]
    # n=7: [0, 1, 4, 10, 12, 17, 20]
    # n=8: [0, 1, 4, 10, 12, 17, 21, 25]
    # n=9: [0, 1, 4, 10, 12, 17, 21, 25, 29]
    # n=10: [0, 1, 4, 10, 12, 17, 21, 25, 29, 34]
    
    # We will run SA for each n.
    # State: list of integers U, sorted, U[0]=0.
    # Moves: 
    # 1. Pick a random index i (1..n-1), change U[i] by +/- delta.
    # 2. Ensure U is sorted and distinct.
    # 3. Ensure U[i] > U[i-1] and (i==n-1 or U[i] < U[i+1]).
    
    # We also need to consider that scaling the whole ruler might help.
    # If we multiply by k, max_U becomes k*max_U_old.
    # |U-U| and |U+U| scale by k, but their SIZES (cardinalities) stay the same 
    # (unless collisions change, which they don't if we just scale).
    # So Ratio is invariant under scaling.
    # Denominator log(2 k max_U + 1) increases with k.
    # So we want k=1 (minimal diameter) for a fixed "shape".
    # Therefore, we just need to find the best "shape" (set of integers with gcd 1, minimal max)
    # for each n.
    
    # Wait, is it always better to have minimal diameter?
    # Score = 1 + log(Ratio) / log(2 max_U + 1).
    # If Ratio > 1, log(Ratio) > 0.
    # We want to MINIMIZE log(2 max_U + 1).
    # So yes, for a fixed set of relative positions (shape), the smallest integers are best.
    # This implies we should search for "optimal" or "near-optimal" Golomb rulers
    # which have the smallest possible diameter for a given n.
    
    # However, maybe a NON-Golomb ruler has a better Ratio?
    # Golomb rulers maximize |U-U|.
    # Do they minimize |U+U|?
    # For U={0,1,4,10}, |S|=10, |D|=13. Ratio 1.3.
    # Try U={0,1,2,5}.
    # S: 0,1,2,3,4,5,6,7,10 -> {0,1,2,3,4,5,6,7,10} size 9.
    # D: -5,-4,-3,-2,-1,0,1,2,3,4,5 -> size 11.
    # Ratio = 11/9 = 1.22.
    # max_U = 5. log(11) = 2.39.
    # Score = 1 + log(1.22)/2.39 = 1 + 0.198/2.39 = 1.083.
    # Golomb {0,1,4,10} score 1.086.
    # Golomb is better here.
    
    # Try U={0,2,3,5} (scaled/shifted?)
    # S: 0,2,3,4,5,6,7,8,10 -> size 9.
    # D: -5,-3,-2,-1,0,1,2,3,5 -> size 9.
    # Ratio = 1.
    # Score = 1.
    
    # It seems Golomb rulers are good candidates.
    # We will run SA to find the best Golomb ruler (or near-Golomb) for each n.
    # Objective: Maximize Score.
    # Since Score depends on Ratio and max_U, and Ratio is invariant under scaling,
    # we can just search for the integer set with minimal max_U for a given "collision pattern".
    # Actually, simpler: Just run SA on the integer values directly, 
    # starting from a known Golomb ruler, and let it find better structures if they exist.
    
    for n in range(3, 21):
        if time.time() > deadline:
            break
            
        # Initialize with a known Golomb ruler for size n
        # We can generate a simple one: 0, 1, 3, 6, 10... (triangular numbers)
        # Triangular numbers T_k = k(k+1)/2.
        # Differences are sums of consecutive integers, not necessarily distinct.
        # Better to use a known list or a greedy construction.
        
        # Greedy Golomb ruler construction:
        U = [0]
        for i in range(1, n):
            # Find smallest x > U[-1] such that all new differences are unique
            # and not in existing differences
            existing_diffs = set()
            for a in U:
                for b in U:
                    existing_diffs.add(abs(a - b))
            
            candidate = U[-1] + 1
            while True:
                # Check if candidate creates duplicate differences
                new_diffs_ok = True
                for a in U:
                    d = abs(a - candidate)
                    if d in existing_diffs and d != 0: # 0 is always there
                        # Wait, 0 is not in existing_diffs if we only add non-zero?
                        # Let's just check all diffs.
                        pass
                    # Actually, simpler:
                    # Construct new set, check if len(diffset) == n(n-1)+1
                    pass
                
                # This greedy is O(n^3) per step. For n=20, it's fine.
                # Let's implement a simple check.
                test_U = U + [candidate]
                test_diffs = set()
                for a in test_U:
                    for b in test_U:
                        test_diffs.add(a - b)
                if len(test_diffs) == n * (n - 1) + 1:
                    break
                candidate += 1
            
            U.append(candidate)
        
        # Now U is a valid Golomb ruler.
        # Run SA to see if we can improve the score.
        # Moves: 
        # 1. Shift one element.
        # 2. Scale? No, scaling reduces score.
        # 3. Maybe "un-Golomb" it? i.e. allow some difference collisions if it significantly reduces |S| or max_U?
        #    But reducing max_U is hard if we keep n fixed.
        #    Actually, if we allow difference collisions, |D| drops.
        #    If |D| drops but |S| drops MORE, Ratio increases.
        #    Is that possible?
        #    Usually |D| >= |S|.
        #    If we collapse some differences, we might also collapse sums.
        
        # Let's just run SA on the current U to see if local improvements exist.
        # We'll use a simple random perturbation.
        
        current_U = U[:]
        current_score = compute_score(current_U)
        
        T = 100.0  # Initial temperature
        cooling = 0.95
        
        # Time limit per n: 2 seconds
        n_start = time.time()
        
        while T > 1.0 and time.time() - n_start < 2.0:
            # Pick a random element to move (not 0)
            idx = random.randint(1, n - 1)
            old_val = current_U[idx]
            
            # Move by random delta
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Ensure validity: sorted, distinct, non-negative
            if new_val < 0:
                T *= cooling
                continue
            
            # Check if new_val is already in U (excluding idx)
            if new_val in current_U and current_U.index(new_val) != idx:
                T *= cooling
                continue
            
            # Construct new list
            new_U = current_U[:]
            new_U[idx] = new_val
            new_U.sort()
            
            # Check if 0 is still 0 (it is, we didn't move idx 0)
            # Check if sorted order is maintained?
            # We sorted it, so yes.
            # But we must ensure U[0] == 0.
            if new_U[0] != 0:
                T *= cooling
                continue
            
            # Check distinctness
            if len(set(new_U)) != n:
                T *= cooling
                continue
            
            new_score = compute_score(new_U)
            
            # Metropolis criterion
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
            else:
                # Accept with probability exp((new_score - current_score) / T)
                # Note: scores are close to 1, so differences are small.
                # We need to scale the temperature or the difference.
                # Let's use a relative difference or a smaller T.
                # Actually, T=100 is huge for scores around 1.
                # Let's use T in the range of 0.01 to 0.001.
                # And scale the delta score.
                pass # This logic is flawed for scores ~1.
            
            T *= cooling
        
        # The SA above is poorly tuned for scores ~1.
        # Let's restart with a better SA strategy.
        
        # Better SA:
        # Objective: Maximize Score.
        # Move: Change one element by +/- 1, 2, or 3.
        # Accept if better, or with prob exp((S_new - S_old) / T).
        # T starts at 0.01 and cools to 0.001.
        
        current_U = U[:]
        current_score = compute_score(current_U)
        
        T = 0.05
        cooling = 0.99
        n_start = time.time()
        
        while T > 0.001 and time.time() - n_start < 3.0:
            idx = random.randint(1, n - 1)
            old_val = current_U[idx]
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = old_val + delta
            
            if new_val < 0:
                T *= cooling
                continue
            
            # Check collision
            valid = True
            for i, v in enumerate(current_U):
                if i != idx and v == new_val:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            new_U = current_U[:]
            new_U[idx] = new_val
            new_U.sort()
            
            if new_U[0] != 0:
                T *= cooling
                continue
            
            new_score = compute_score(new_U)
            
            if new_score >= current_score:
                current_U = new_U
                current_score = new_score
            else:
                # Calculate acceptance probability
                # Delta is small, T is small.
                try:
                    prob = math.exp((new_score - current_score) / T)
                    if random.random() < prob:
                        current_U = new_U
                        current_score = new_score
                except:
                    pass
            
            T *= cooling
        
        if current_score > best_score:
            best_score = current_score
            best_U = current_U[:]
            
    # After searching all n, return the best U found.
    if best_U is None or len(best_U) == 0:
        return [0, 1, 2]
        
    return best_U

# EVOLVE-BLOCK-END
