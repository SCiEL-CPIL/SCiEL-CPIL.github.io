import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_sums = list(range(target_lo, target_hi + 1))
    
    def compute_multiplicities(A):
        """Compute representation multiplicity for each target sum."""
        A_set = set(A)
        mults = {s: 0 for s in target_sums}
        A_sorted = sorted(A)
        for i in range(len(A_sorted)):
            for j in range(i, len(A_sorted)):
                s = A_sorted[i] + A_sorted[j]
                if target_lo <= s <= target_hi:
                    mults[s] += 1
        return mults
    
    def peak_mult(mults):
        return max(mults.values())
    
    def is_valid(A):
        """Check if all target sums have at least one representation."""
        A_set = set(A)
        for s in target_sums:
            found = False
            for a in A:
                b = s - a
                if b in A_set and a <= b:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Initialization: try a structured set
    # Start with all even numbers from 0 to 128
    A_init = list(range(0, 129, 2))  # 65 elements: 0,2,4,...,128
    
    # Try a smaller structured set: spread out
    # Use a "balanced" approach
    random.seed(42)
    
    best_A = None
    best_peak = float('inf')
    
    # Try several initializations
    initializations = []
    
    # Init 1: Even numbers
    initializations.append(list(range(0, 129, 2)))
    
    # Init 2: All numbers (too big, but check)
    # initializations.append(list(range(0, 129)))
    
    # Init 3: Random subset of size ~40
    for trial in range(5):
        size = 35 + trial * 3
        A_rand = random.sample(range(0, 129), size)
        initializations.append(A_rand)
    
    # Init 4: Structured - two clusters
    # Low cluster and high cluster
    A_struct = list(range(0, 20)) + list(range(109, 129))
    initializations.append(A_struct)
    
    # Init 5: Spread out evenly
    A_spread = [int(i * 128 / 30) for i in range(31)]
    A_spread = sorted(set(A_spread))
    initializations.append(A_spread)
    
    # Evaluate initializations
    for A_init in initializations:
        if time.time() > deadline - 10.0:
            break
        if not A_init or len(A_init) < 2:
            continue
        A_set = set(A_init)
        A_sorted = sorted(A_set)
        if not is_valid(A_sorted):
            continue
        mults = compute_multiplicities(A_sorted)
        peak = peak_mult(mults)
        if peak < best_peak:
            best_peak = peak
            best_A = A_sorted[:]
    
    if best_A is None:
        # Fallback: use the seed
        return list(range(64, 129))
    
    # Local search: try to improve
    A = best_A[:]
    current_peak = best_peak
    A_set = set(A)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        # Try random moves: swap an element in/out, or add/remove
        move_type = random.choice(['swap', 'remove', 'add'])
        
        if move_type == 'swap' and len(A) > 1:
            i = random.randint(0, len(A) - 1)
            old_val = A[i]
            new_val = random.randint(0, n)
            if new_val in A_set:
                continue
            # Try the swap
            A[i] = new_val
            A_set.remove(old_val)
            A_set.add(new_val)
            A_sorted = sorted(A)
            
            if is_valid(A_sorted):
                mults = compute_multiplicities(A_sorted)
                peak = peak_mult(mults)
                if peak < current_peak:
                    current_peak = peak
                    no_improve_count = 0
                    best_A = A_sorted[:]
                    best_peak = current_peak
                else:
                    # Revert
                    A[i] = old_val
                    A_set.remove(new_val)
                    A_set.add(old_val)
                    no_improve_count += 1
            else:
                # Revert
                A[i] = old_val
                A_set.remove(new_val)
                A_set.add(old_val)
                no_improve_count += 1
        
        elif move_type == 'remove' and len(A) > 10:
            i = random.randint(0, len(A) - 1)
            old_val = A[i]
            A.pop(i)
            A_set.remove(old_val)
            A_sorted = sorted(A)
            
            if is_valid(A_sorted):
                mults = compute_multiplicities(A_sorted)
                peak = peak_mult(mults)
                if peak < current_peak:
                    current_peak = peak
                    no_improve_count = 0
                    best_A = A_sorted[:]
                    best_peak = current_peak
                else:
                    # Revert
                    A.append(old_val)
                    A_set.add(old_val)
                    A.sort()
                    no_improve_count += 1
            else:
                # Revert
                A.append(old_val)
                A_set.add(old_val)
                A.sort()
                no_improve_count += 1
        
        elif move_type == 'add' and len(A) < 100:
            new_val = random.randint(0, n)
            if new_val in A_set:
                continue
            A.append(new_val)
            A_set.add(new_val)
            A_sorted = sorted(A)
            
            mults = compute_multiplicities(A_sorted)
            peak = peak_mult(mults)
            if peak < current_peak:
                current_peak = peak
                no_improve_count = 0
                best_A = A_sorted[:]
                best_peak = current_peak
            else:
                # Revert
                A.pop()
                A_set.remove(new_val)
                no_improve_count += 1
    
    return best_A
