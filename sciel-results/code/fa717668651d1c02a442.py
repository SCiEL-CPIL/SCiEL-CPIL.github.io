import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    targets = list(range(target_lo, target_hi + 1))
    num_targets = len(targets)  # 129
    
    # Helper: compute representation counts for a given set A
    def compute_reps(A):
        A_set = set(A)
        reps = [0] * num_targets
        # For each unordered pair (a, b) with a <= b
        sorted_A = sorted(A)
        k = len(sorted_A)
        for i in range(k):
            a = sorted_A[i]
            for j in range(i, k):
                b = sorted_A[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    reps[s - target_lo] += 1
        return reps
    
    def peak_mult(A):
        reps = compute_reps(A)
        return max(reps)
    
    def is_valid(A):
        reps = compute_reps(A)
        return all(r > 0 for r in reps)
    
    # Initial solution: try a structured approach
    # Strategy: use a set with good sum coverage
    # Start with the baseline and try to improve
    
    # Try several initial structures
    candidates = []
    
    # 1. Baseline: [64, 128]
    candidates.append(list(range(64, 129)))
    
    # 2. Spread out: use every other element in a wider range
    #    e.g., {0, 2, 4, ..., 128} - but this might not cover all sums
    #    Let's think: if A = {0, 3, 6, ..., 126} (43 elements, step 3)
    #    Sums range from 0 to 252, but gaps exist
    
    # 3. Use a "dense" core with some spread
    #    Try: A = [x for x in range(0, 129) if x % 2 == 0] -> 65 elements, even numbers
    #    Sums of two evens are even, so odd targets in [128,256] won't be covered
    #    So we need both parities
    
    # 4. Mixed: some structure
    #    Let's try: A = list(range(0, 129, 2)) + list(range(1, 129, 4))
    #    That's 65 + 32 = 97 elements, too many
    
    # Better: start with baseline and do local search
    best_A = list(range(64, 129))
    best_peak = peak_mult(best_A)
    
    # Local search: try swapping elements in/out
    def local_search(A, max_iters=5000):
        A = list(A)
        current_peak = peak_mult(A)
        best_A_local = list(A)
        best_peak_local = current_peak
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            new_A = list(A)
            
            if move_type < 0.3 and len(new_A) < 129:
                # Add a random element
                candidates_to_add = [x for x in range(129) if x not in set(new_A)]
                if candidates_to_add:
                    new_A.append(random.choice(candidates_to_add))
            elif move_type < 0.5 and len(new_A) > 1:
                # Remove a random element
                idx = random.randrange(len(new_A))
                new_A.pop(idx)
            else:
                # Swap: remove one, add one
                if len(new_A) > 1:
                    idx = random.randrange(len(new_A))
                    removed = new_A.pop(idx)
                    candidates_to_add = [x for x in range(129) if x not in set(new_A)]
                    if candidates_to_add:
                        new_A.append(random.choice(candidates_to_add))
            
            if not new_A:
                continue
            
            new_peak = peak_mult(new_A)
            
            # Accept if valid and better, or with some probability if worse
            if is_valid(new_A):
                if new_peak < best_peak_local:
                    best_peak_local = new_peak
                    best_A_local = list(new_A)
                    A = new_A
                    current_peak = new_peak
                elif new_peak <= current_peak + 2:
                    # Accept worse solutions sometimes for exploration
                    if random.random() < 0.3:
                        A = new_A
                        current_peak = new_peak
            else:
                # Invalid: don't accept unless it's much better in some other sense
                # Just skip
                pass
        
        return best_A_local, best_peak_local
    
    # Run local search with multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            A_start = list(best_A)
        else:
            # Perturb the best solution
            A_start = list(best_A)
            # Remove some random elements and add others
            for _ in range(random.randint(5, 15)):
                if len(A_start) > 1:
                    idx = random.randrange(len(A_start))
                    removed = A_start.pop(idx)
                    candidates_to_add = [x for x in range(129) if x not in set(A_start)]
                    if candidates_to_add:
                        A_start.append(random.choice(candidates_to_add))
        
        A_new, peak_new = local_search(A_start, max_iters=3000)
        
        if is_valid(A_new) and peak_new < best_peak:
            best_peak = peak_new
            best_A = A_new
    
    # Final check
    if not is_valid(best_A):
        # Fallback to baseline
        return list(range(64, 129))
    
    return best_A
