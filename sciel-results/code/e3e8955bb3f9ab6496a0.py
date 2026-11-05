import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55  # 55 second deadline, 5s safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = range(target_lo, target_hi + 1)
    
    def compute_representations(A):
        """Compute representation counts for each target sum."""
        A_set = set(A)
        counts = {}
        for s in target_range:
            cnt = 0
            for a in A:
                b = s - a
                if b in A_set and a <= b:
                    cnt += 1
            counts[s] = cnt
        return counts
    
    def is_valid(A):
        """Check if all target sums are covered."""
        A_set = set(A)
        for s in target_range:
            found = False
            for a in A:
                b = s - a
                if b in A_set and a <= b:
                    found = True
                    break
            if not found:
                return False
        return True
    
    def peak_mult(A):
        """Compute the peak representation multiplicity."""
        A_set = set(A)
        peak = 0
        for s in target_range:
            cnt = 0
            for a in A:
                b = s - a
                if b in A_set and a <= b:
                    cnt += 1
            if cnt > peak:
                peak = cnt
        return peak
    
    def evaluate(A):
        """Return (peak, is_valid)."""
        A_set = set(A)
        peak = 0
        valid = True
        for s in target_range:
            cnt = 0
            for a in A:
                b = s - a
                if b in A_set and a <= b:
                    cnt += 1
            if cnt == 0:
                valid = False
            if cnt > peak:
                peak = cnt
        return peak, valid
    
    # Start with baseline
    best_A = list(range(64, 129))
    best_peak, _ = evaluate(best_A)
    
    # Try local search
    A = list(best_A)
    current_peak, _ = evaluate(A)
    
    # Local search: try swapping elements
    all_candidates = list(range(0, 129))
    
    while time.time() < deadline:
        improved = False
        # Try all possible swaps and adds/removes
        for i in range(len(A)):
            if time.time() > deadline:
                break
            for x in all_candidates:
                if x in A:
                    continue
                # Try swapping A[i] with x
                new_A = A.copy()
                new_A[i] = x
                new_A = list(set(new_A))  # ensure unique
                if len(new_A) != len(A):
                    continue
                new_peak, new_valid = evaluate(new_A)
                if new_valid and new_peak < current_peak:
                    A = new_A
                    current_peak = new_peak
                    improved = True
                    break
            if improved:
                break
            if time.time() > deadline:
                break
        
        if not improved:
            # Try adding an element
            added = False
            for x in all_candidates:
                if time.time() > deadline:
                    break
                if x in A:
                    continue
                new_A = A + [x]
                new_peak, new_valid = evaluate(new_A)
                if new_valid and new_peak < current_peak:
                    A = new_A
                    current_peak = new_peak
                    added = True
                    break
            if not added:
                # Try removing an element
                removed = False
                for i in range(len(A)):
                    if time.time() > deadline:
                        break
                    new_A = A[:i] + A[i+1:]
                    if len(new_A) == 0:
                        continue
                    new_peak, new_valid = evaluate(new_A)
                    if new_valid and new_peak < current_peak:
                        A = new_A
                        current_peak = new_peak
                        removed = True
                        break
            if not removed and not added:
                # No improvement possible with single moves, try random restart
                # Generate a random valid set
                for attempt in range(10):
                    if time.time() > deadline:
                        break
                    # Random subset of size between 20 and 65
                    size = random.randint(20, 65)
                    rand_A = list(random.sample(all_candidates, size))
                    rand_peak, rand_valid = evaluate(rand_A)
                    if rand_valid and rand_peak < current_peak:
                        A = rand_A
                        current_peak = rand_peak
                        break
        
        if current_peak < best_peak:
            best_A = A.copy()
            best_peak = current_peak
    
    return best_A
