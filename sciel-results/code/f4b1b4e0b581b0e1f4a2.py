import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 50.0  # Keep well under 60s/90s limit
    
    n = 128
    target_low = 128
    target_high = 256
    target_range = list(range(target_low, target_high + 1))
    
    # Initial set: [64, 129] covers [128, 256]
    # 64+64=128, 128+128=256
    # This is a valid solution.
    A = set(range(64, 129))
    
    def calc_counts(A_set):
        # Returns an array of counts for sums in [0, 256]
        # A is a set of ints
        counts = np.zeros(257, dtype=np.int32)
        A_list = list(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 256:
                    counts[s] += 1
        return counts

    def get_peak(counts):
        return np.max(counts[target_low:target_high+1])

    def is_valid(counts):
        return np.min(counts[target_low:target_high+1]) > 0

    # Initial state
    counts = calc_counts(A)
    current_peak = get_peak(counts)
    best_A = set(A)
    best_peak = current_peak
    
    # Local Search
    # We will try to swap elements or add/remove to reduce peak.
    # Objective: minimize peak, subject to min count >= 1.
    
    # To speed up, we can use a greedy approach first:
    # 1. Try to remove elements that don't break validity and reduce peak.
    # 2. Try to swap elements to reduce peak.
    
    # Let's use a simple local search with restarts or a single long run.
    # Given the small search space, a single run with simulated annealing or hill climbing should work.
    
    # We'll use a hill climbing approach with random restarts if stuck, or just a long hill climb.
    # Let's try a greedy removal first to shrink the set.
    
    # Greedy Removal
    changed = True
    while changed:
        changed = False
        # Sort by contribution to peak? Or just iterate.
        for x in list(A):
            if len(A) <= 1: break
            A_test = set(A)
            A_test.remove(x)
            counts_test = calc_counts(A_test)
            if is_valid(counts_test):
                peak_test = get_peak(counts_test)
                if peak_test < current_peak or (peak_test == current_peak and len(A_test) < len(A)):
                    A = A_test
                    counts = counts_test
                    current_peak = peak_test
                    changed = True
                    if len(A) < len(best_A) or (len(A) == len(best_A) and current_peak < best_peak):
                        best_A = set(A)
                        best_peak = current_peak
                    break # Restart iteration after change
        
    # Now Local Search: Swap elements
    # We want to minimize peak.
    # State: A, counts, current_peak
    # Move: x in A, y not in A.
    
    # To make this efficient, we can precompute or just iterate.
    # |A| is likely small (e.g., < 20).
    
    # Let's do a random local search
    # Temperature annealing or just random moves that improve or maintain peak?
    # Minimize peak. If peak decreases, accept. If same peak, maybe accept to escape local minima?
    # Let's try: Accept if peak decreases. If same, accept with prob 0.5 (to shuffle).
    
    # Actually, minimizing peak is the primary goal.
    # We can also try to minimize the sum of squares of counts to flatten the distribution.
    # Objective: lexicographic (peak, sum_of_squares)
    
    def objective(counts):
        peak = get_peak(counts)
        # Sum of squares of counts in target range
        ss = np.sum(counts[target_low:target_high+1] ** 2)
        return (peak, ss)
    
    current_obj = objective(counts)
    best_obj = current_obj
    best_A = set(A)
    
    # Local Search Loop
    while time.time() - start_time < timeout:
        # Random move: swap x in A with y not in A
        if len(A) == 0:
            A.add(128)
            counts = calc_counts(A)
            continue
            
        x = random.choice(list(A))
        # Choose y from [0, 128] \ A
        # To avoid slow sampling, we can just pick a random number and retry if in A
        y = random.randint(0, 128)
        if y in A:
            continue
        
        A_test = set(A)
        A_test.remove(x)
        A_test.add(y)
        
        counts_test = calc_counts(A_test)
        if not is_valid(counts_test):
            continue
            
        obj_test = objective(counts_test)
        
        # Accept if better
        if obj_test < current_obj:
            A = A_test
            counts = counts_test
            current_obj = obj_test
            if obj_test < best_obj:
                best_obj = obj_test
                best_A = set(A)
        # Accept with small prob if same peak but lower SS?
        # Or just strictly improve.
        # Let's add a bit of randomness to escape plateaus
        elif obj_test[0] == current_obj[0] and obj_test[1] < current_obj[1]:
             A = A_test
             counts = counts_test
             current_obj = obj_test
             if obj_test < best_obj:
                 best_obj = obj_test
                 best_A = set(A)
        elif obj_test[0] == current_obj[0] and random.random() < 0.1:
             # Random walk on same peak to find better SS
             A = A_test
             counts = counts_test
             current_obj = obj_test
             if obj_test < best_obj:
                 best_obj = obj_test
                 best_A = set(A)
        
        # Also try adding an element if it reduces peak? 
        # Unlikely to reduce peak, but might allow removing others later.
        # And removing an element?
        # The swap covers most cases.
        
    return list(best_A)

# Test locally
if __name__ == "__main__":
    A = construct()
    # Verify
    counts = np.zeros(257, dtype=np.int32)
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            aj = A[j]
            s = ai + aj
            if s <= 256:
                counts[s] += 1
    peak = np.max(counts[128:257])
    min_cov = np.min(counts[128:257])
    print(f"Size: {len(A)}, Peak: {peak}, Min Cov: {min_cov}")
    print(f"Set: {sorted(A)}")
