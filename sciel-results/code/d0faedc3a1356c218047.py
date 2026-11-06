import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = range(target_lo, target_hi + 1)
    num_targets = len(target_range)
    
    best_A = None
    best_score = (float('inf'), float('inf')) # (uncovered, peak)
    
    def calculate_counts(A):
        """Calculate representation counts for all sums in [128, 256]."""
        counts = [0] * (target_hi + 1)
        A_list = list(A)
        k = len(A_list)
        for i in range(k):
            a = A_list[i]
            for j in range(i, k):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def get_score(counts):
        """Return (uncovered_count, max_representation_count)."""
        uncovered = 0
        peak = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            elif c > peak:
                peak = c
        return (uncovered, peak)

    def local_search(initial_A, max_steps=20000):
        """
        Perform local search starting from initial_A.
        Returns the best A found and its score.
        """
        A = list(initial_A)
        A_set = set(A)
        counts = calculate_counts(A)
        curr_score = get_score(counts)
        
        best_A_local = A[:]
        best_score_local = curr_score
        
        no_improve_steps = 0
        stuck_threshold = 2000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # If stuck for too long, break to allow ILS to perturb
            if no_improve_steps > stuck_threshold:
                break
                
            # Select a random element x to remove
            if not A:
                break
            x = random.choice(A)
            
            # Generate candidate y to add
            # To improve efficiency, we can bias y towards values that might cover uncovered sums
            # But for simplicity and robustness, random y with retries is fine.
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 10:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                continue

            # Apply move: Remove x, Add y
            # We need to update counts efficiently.
            # Old pairs involving x: (x, x) and (x, z) for z in A, z != x
            # New pairs involving y: (y, y) and (y, z) for z in A (after removing x, but y is not in A yet)
            
            # Let's do a safe copy-and-apply to avoid complex rollback logic errors, 
            # but we can optimize by just updating the counts array directly and reverting if bad.
            
            # Save state
            old_A = A[:]
            old_counts = counts[:]
            
            # Remove x
            A.remove(x)
            A_set.remove(x)
            
            # Update counts for removal of x
            # Pairs (x, x)
            s_xx = 2 * x
            if target_lo <= s_xx <= target_hi:
                counts[s_xx] -= 1
            # Pairs (x, z)
            for z in A:
                s_xz = x + z
                if target_lo <= s_xz <= target_hi:
                    counts[s_xz] -= 1
            
            # Add y
            A.append(y)
            A_set.add(y)
            
            # Update counts for addition of y
            # Pairs (y, y)
            s_yy = 2 * y
            if target_lo <= s_yy <= target_hi:
                counts[s_yy] += 1
            # Pairs (y, z)
            for z in A:
                if z == y:
                    continue
                s_yz = y + z
                if target_lo <= s_yz <= target_hi:
                    counts[s_yz] += 1
            
            new_score = get_score(counts)
            
            if new_score < best_score_local:
                best_score_local = new_score
                best_A_local = A[:]
                no_improve_steps = 0
            elif new_score < curr_score:
                # Accept move if it improves current, even if not global best
                curr_score = new_score
                no_improve_steps = 0
            else:
                # Rollback
                A = old_A
                A_set = set(A)
                counts = old_counts
                no_improve_steps += 1
                
        return best_A_local, best_score_local

    def perturb(A, num_changes):
        """Randomly change num_changes elements in A."""
        A_perturbed = list(A)
        A_set = set(A)
        for _ in range(num_changes):
            if not A_perturbed:
                break
            x = random.choice(A_perturbed)
            A_perturbed.remove(x)
            A_set.remove(x)
            
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 10:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                # If we can't find a new y, just put x back
                A_perturbed.append(x)
                A_set.add(x)
                continue
            
            A_perturbed.append(y)
            A_set.add(y)
        return A_perturbed

    # Main Loop: Iterated Local Search
    # Try different initial sizes and strategies
    
    # Strategy 1: Random initialization with various sizes
    # Strategy 2: Structured initialization
    
    initializations = []
    
    # Random sets
    for k in [30, 40, 50, 60]:
        for _ in range(3):
            initializations.append(random.sample(range(n + 1), k))
            
    # Structured: Arithmetic progressions or spread out
    # e.g., 0, 3, 6, ...
    for step_size in [3, 4, 5]:
        A_struct = list(range(0, n + 1, step_size))
        if len(A_struct) > 0:
            initializations.append(A_struct)
            
    # Mix of low and high
    for _ in range(3):
        low_part = random.sample(range(0, 64), 15)
        high_part = random.sample(range(64, 129), 15)
        initializations.append(low_part + high_part)

    # Shuffle initializations to ensure variety
    random.shuffle(initializations)
    
    for init_A in initializations:
        if time.time() > deadline:
            break
            
        # Run local search
        A_res, score_res = local_search(init_A, max_steps=15000)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
            
        # If we found a perfect solution (0 uncovered, low peak), we might stop early
        # But we want to minimize peak, so keep going if peak > 3?
        # The theoretical minimum peak for this problem is likely small (2 or 3).
        if best_score[0] == 0 and best_score[1] <= 3:
            # Try to improve further? Or break?
            # Let's do a few more ILS iterations to see if we can get peak 2.
            pass

    # If we still have time, do some ILS on the best solution found
    if best_A is not None and time.time() < deadline - 5.0:
        current_best_A = best_A
        current_best_score = best_score
        
        for i in range(20): # Max 20 ILS iterations
            if time.time() > deadline:
                break
            
            # Perturb the current best
            # Number of changes: proportional to size
            num_changes = max(2, len(current_best_A) // 4)
            perturbed_A = perturb(current_best_A, num_changes)
            
            # Local search from perturbed state
            A_res, score_res = local_search(perturbed_A, max_steps=10000)
            
            if score_res < current_best_score:
                current_best_score = score_res
                current_best_A = A_res
                best_score = current_best_score
                best_A = current_best_A
            else:
                # Keep the best found so far as the base for next perturbation
                # Or revert to the original best? 
                # Usually, ILS keeps the best found globally.
                pass
                
    if best_A is None:
        # Fallback: A set that definitely covers the range, e.g., [0..128]
        # This will have high peak, but it's valid.
        return list(range(n + 1))
        
    return best_A
