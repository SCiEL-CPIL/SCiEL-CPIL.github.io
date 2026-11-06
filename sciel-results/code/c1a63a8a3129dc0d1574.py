import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    target_min = 128
    target_max = 256
    
    # Pre-compute target range size
    target_size = target_max - target_min + 1
    
    def get_counts(A_set):
        """
        Compute the representation counts for all targets in [128, 256].
        Returns an array of length 129 (indices 128..256) where count[t] is 
        the number of unordered pairs (a,b) with a<=b, a,b in A, a+b=t.
        """
        counts = np.zeros(target_size, dtype=np.int32)
        A_list = list(A_set)
        # Optimization: if A is small, direct O(|A|^2) is fine.
        # For |A| ~ 60, |A|^2 ~ 3600. This is very fast.
        for i, a in enumerate(A_list):
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_min <= s <= target_max:
                    counts[s - target_min] += 1
        return counts

    def evaluate(A_set):
        """
        Returns (peak_multiplicity, is_valid, size)
        """
        counts = get_counts(A_set)
        if np.any(counts == 0):
            return 10000, False, len(A_set) # Invalid
        peak = np.max(counts)
        return peak, True, len(A_set)

    # 1. Greedy Initialization
    # Start with a set that is likely to cover everything.
    # A simple heuristic: take every k-th number? 
    # Or just start with all numbers and prune? 
    # Let's try a random dense start and prune, which is robust.
    
    best_A = None
    best_peak = 10000
    best_size = 10000
    
    # We will run multiple restarts of the ILS process
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize A
        if restart == 0:
            # Start with a structured set: even numbers? 
            # [0, 2, 4, ..., 128] covers sums [0, 256] with step 2.
            # Sums of evens are even. We need odd sums too.
            # So we need at least one odd number.
            # Let's start with all numbers 0..128 and prune.
            A = set(range(0, 129))
        else:
            # Random dense set
            A = set(random.sample(range(0, 129), random.randint(80, 129)))
            
        # Ensure validity by adding missing coverage if needed?
        # For now, assume we prune from a valid set.
        # Check if initial A is valid.
        peak, valid, size = evaluate(A)
        
        if not valid:
            # Try to fix by adding random elements until valid
            attempts = 0
            while not valid and attempts < 100:
                x = random.randint(0, 128)
                if x not in A:
                    A.add(x)
                    peak, valid, size = evaluate(A)
                attempts += 1
            if not valid:
                continue
                
        current_A = A
        current_peak = peak
        current_size = size
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to remove elements
            # We want to remove x if it keeps all counts >= 1.
            # To optimize peak, we prefer removing x that doesn't increase the peak.
            
            removable_candidates = []
            counts = get_counts(current_A)
            
            for x in current_A:
                # Check if removing x keeps validity
                # We can simulate removal quickly
                # But get_counts is O(|A|^2). 
                # Better: calculate delta for removing x.
                # Removing x decreases count[t] for all t = x+y (y in A, y>=x) and t = x+y (y in A, y<x).
                # Actually, unordered pairs involving x:
                # (x, y) for y in A, y >= x.
                # The count for sum s decreases by 1 for each such pair.
                
                # Let's just do the full evaluation for simplicity first, 
                # but cache counts if possible? 
                # Given |A| <= 129, |A|^2 ~ 16k. 
                # 129 elements * 16k ops = 2M ops per sweep. 
                # Python is slow. 2M ops ~ 0.1-0.2s. 
                # 100 sweeps ~ 10-20s. This might be too slow for 20 restarts.
                
                # Optimization: Precompute neighbors?
                # Let's stick to a smarter removal check.
                
                # For a specific x, find all sums s = x+y where y in A, y>=x.
                # If any count[s] == 1, we cannot remove x (would make it 0).
                # If all count[s] > 1, we can remove x.
                # New peak after removal:
                # The peak might decrease if x was involved in the current max count sums.
                # Or it might stay the same.
                
                can_remove = True
                new_peak = current_peak
                # We need to know the new counts.
                # Let's compute the delta.
                # This is getting complex. Let's just use the full evaluation 
                # but limit the number of iterations per restart.
                
                # Alternative: Use numpy vectorization for the check.
                # This is tricky with sets.
                
                # Let's just do the naive check but break early if time is tight.
                if time.time() > deadline - 5:
                    break
                    
                new_A = current_A - {x}
                p, v, s = evaluate(new_A)
                if v:
                    # Accept if peak is lower, or peak is same and size is smaller
                    if p < current_peak or (p == current_peak and s < current_size):
                        current_A = new_A
                        current_peak = p
                        current_size = s
                        improved = True
                        # Update global best
                        if current_peak < best_peak or (current_peak == best_peak and current_size < best_size):
                            best_peak = current_peak
                            best_size = current_size
                            best_A = set(current_A)
                        # Restart the inner loop to re-evaluate removals from new state
                        break 
                        
            if improved:
                continue
                
            # Phase 2: Try Swaps (Remove x, Add y)
            # This allows us to move to a different local optimum.
            # Try random swaps.
            swap_improved = False
            for _ in range(50): # Limit swaps per local search phase
                if time.time() > deadline:
                    break
                if len(current_A) == 0:
                    break
                x = random.choice(list(current_A))
                y = random.randint(0, 128)
                if x == y:
                    continue
                new_A = current_A - {x}
                new_A.add(y)
                p, v, s = evaluate(new_A)
                if v:
                    if p < current_peak or (p == current_peak and s < current_size):
                        current_A = new_A
                        current_peak = p
                        current_size = s
                        improved = True
                        swap_improved = True
                        if current_peak < best_peak or (current_peak == best_peak and current_size < best_size):
                            best_peak = current_peak
                            best_size = current_size
                            best_A = set(current_A)
                        break # Restart loop after improvement
            
            if swap_improved:
                continue
                
            # If no improvement, perturb
            # Randomly flip 5 bits
            for _ in range(5):
                x = random.randint(0, 128)
                if x in current_A:
                    current_A.remove(x)
                else:
                    current_A.add(x)
            
            # Re-evaluate after perturbation
            current_peak, valid, current_size = evaluate(current_A)
            if not valid:
                # Try to repair
                attempts = 0
                while not valid and attempts < 50:
                    y = random.randint(0, 128)
                    current_A.add(y)
                    current_peak, valid, current_size = evaluate(current_A)
                    attempts += 1
                if not valid:
                    break
                    
            # Update best if valid
            if valid and (current_peak < best_peak or (current_peak == best_peak and current_size < best_size)):
                best_peak = current_peak
                best_size = current_size
                best_A = set(current_A)
                
    # Final verification and return
    if best_A is None:
        # Fallback: All numbers
        best_A = set(range(0, 129))
        
    # One last quick check to ensure validity
    p, v, s = evaluate(best_A)
    if not v:
        best_A = set(range(0, 129))
        
    return sorted(best_A)
