import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin (evaluator kills at 90s)
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    target_len = max_sum - min_sum + 1  # 129
    
    # Precompute valid pairs for each x in [0, 128]
    # valid_pairs[x] is a list of (y, sum) where y in [0, 128] and min_sum <= x+y <= max_sum
    # We only care about sums in [128, 256]
    valid_pairs = {}
    for x in range(N + 1):
        pairs = []
        # y must be in [0, 128]
        # x + y >= 128 => y >= 128 - x
        # x + y <= 256 => y <= 256 - x
        y_min = max(0, 128 - x)
        y_max = min(128, 256 - x)
        for y in range(y_min, y_max + 1):
            pairs.append((y, x + y))
        valid_pairs[x] = pairs

    def compute_counts(A_list):
        """
        Compute counts of sums for target range [128, 256].
        Returns a numpy array of length 129 where index i corresponds to sum 128+i.
        """
        counts = np.zeros(target_len, dtype=np.int32)
        k = len(A_list)
        if k == 0:
            return counts
        
        A_arr = np.array(A_list, dtype=np.int32)
        # Use outer sum
        sums = np.add.outer(A_arr, A_arr)
        
        # We need to count unordered pairs.
        # For each sum s in [128, 256], count the number of pairs (i, j) with i <= j such that A[i]+A[j]=s.
        # Instead of bincounting the whole matrix, we can iterate over the target sums and check.
        # But bincount on the flattened matrix is faster for small k.
        
        # Let's use the property:
        # Total occurrences of s in outer product = 2 * (# pairs with a!=b) + 1 * (# pairs with a==b)
        # Unordered pairs count = (# pairs with a!=b) + (# pairs with a==b)
        # Let f(s) be total occurrences.
        # Let g(s) be number of a in A such that 2a=s.
        # Unordered count = (f(s) + g(s)) / 2.
        
        # Filter sums to target range
        # We only care about sums in [128, 256]
        # Create a mask for the sums matrix
        mask = (sums >= min_sum) & (sums <= max_sum)
        valid_sums = sums[mask]
        
        if len(valid_sums) == 0:
            return counts
            
        # Map sums to indices 0..128
        indices = valid_sums - min_sum
        f = np.bincount(indices, minlength=target_len).astype(np.int32)
        
        # Calculate g(s)
        g = np.zeros(target_len, dtype=np.int32)
        A_set = set(A_list)
        for a in A_list:
            s = 2 * a
            if min_sum <= s <= max_sum:
                g[s - min_sum] += 1
        
        counts = (f + g) // 2
        return counts

    def get_cost(A_list):
        """
        Cost: max multiplicity in target range.
        Penalty if not all sums are covered.
        """
        counts = compute_counts(A_list)
        if np.any(counts == 0):
            uncovered = np.sum(counts == 0)
            return 10000 + uncovered
        return int(np.max(counts))

    def get_smooth_cost(A_list, beta=1.0):
        """
        Smooth surrogate for max multiplicity.
        """
        counts = compute_counts(A_list).astype(np.float64)
        if np.any(counts == 0):
            return 10000.0 + np.sum(counts == 0)
        
        # Log-sum-exp approximation of max
        # max(x) approx = (1/beta) * log(sum(exp(beta*x)))
        # To avoid overflow, subtract max
        max_c = np.max(counts)
        lse = max_c + (1.0 / beta) * np.log(np.sum(np.exp(beta * (counts - max_c))))
        return lse

    # --- Phase 1: Greedy Construction with Pruning ---
    
    A_list = []
    A_set = set()
    current_counts = np.zeros(target_len, dtype=np.int32)
    
    # We need to cover 129 sums.
    # Start with a few random elements or a structured start?
    # A structured start might be better: include 0, 128, and some midpoints.
    # But greedy is robust. Let's start empty and add greedily.
    
    # To help greedy, we can seed with a small random set
    random.seed(42)
    # Seed with 5 random elements
    for _ in range(5):
        x = random.randint(0, N)
        if x not in A_set:
            A_list.append(x)
            A_set.add(x)
    
    # Update counts
    current_counts = compute_counts(A_list)
    
    # Greedy loop: Add elements until all sums are covered
    while True:
        if time.time() > deadline - 5.0:
            break
        
        if not np.any(current_counts == 0):
            break
            
        # Find the best x to add
        best_x = None
        best_gain = -1
        best_max_mult = float('inf')
        
        # We want to maximize coverage of 0s.
        # Secondary objective: minimize the resulting max multiplicity.
        
        uncovered_sums = np.where(current_counts == 0)[0]
        
        for x in range(N + 1):
            if x in A_set:
                continue
            
            # Calculate new counts if we add x
            # New pairs involving x: (x, y) for y in A_list, and (x, x)
            # We need to update counts only for sums in [128, 256]
            
            # Instead of full recompute, we can simulate the change.
            # But full recompute is O(k^2) which is small.
            # Let's just estimate the gain.
            
            # Gain = number of uncovered sums that become covered.
            # A sum s is covered if there exists y in A_list such that x+y=s, or 2x=s.
            
            # For each uncovered sum s:
            #   Check if s-x is in A_list, or 2x==s.
            
            gain = 0
            # To speed up, we can check which uncovered sums are "reachable" by x.
            # s = x + y => y = s - x.
            # If y in A_set, then s is covered by (x, y).
            # If 2x == s, then s is covered by (x, x).
            
            # Let's collect the set of sums that x would cover.
            covered_by_x = set()
            for y in A_list:
                s = x + y
                if min_sum <= s <= max_sum:
                    covered_by_x.add(s - min_sum)
            s_double = 2 * x
            if min_sum <= s_double <= max_sum:
                covered_by_x.add(s_double - min_sum)
            
            # Count how many of these are currently uncovered
            for idx in covered_by_x:
                if current_counts[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                # We can't easily compute the new max mult without full recompute,
                # but we can prioritize higher gain first.
            elif gain == best_gain and gain > 0:
                # Tie-breaker: choose x that results in lower max multiplicity?
                # This is expensive. Let's just pick the first one or random among best.
                # For now, we stick with the first best found.
                pass
        
        if best_x is None or best_gain == 0:
            # If no single addition covers new ground, we might be stuck?
            # Or maybe we need to add an element that doesn't cover *new* ground but helps balance?
            # Actually, if we haven't covered all sums, there must be some x that covers at least one uncovered sum?
            # Not necessarily. If A is such that no new x covers a specific uncovered sum s,
            # it means for all x not in A, s-x not in A and 2x != s.
            # This is possible. In that case, we might need to swap.
            # But let's try adding a random element first to break symmetry.
            candidates = [x for x in range(N + 1) if x not in A_set]
            if not candidates:
                break
            best_x = random.choice(candidates)
            # If gain is 0, we are just expanding A without covering new sums.
            # This might be necessary to enable future coverage?
            # Actually, if gain is 0, adding x doesn't help cover *currently* uncovered sums.
            # It might help cover sums that are currently covered but with high multiplicity? No, that increases mult.
            # It might help cover sums that are *not yet uncovered*? No, all sums are either covered or not.
            # If no x covers any uncovered sum, then the set of uncovered sums is "isolated" from A.
            # This suggests we need to change A significantly (swap).
            
            # For now, if gain is 0, we break greedy and rely on local search to fix.
            if best_gain == 0:
                break
        
        A_list.append(best_x)
        A_set.add(best_x)
        
        # Update counts
        current_counts = compute_counts(A_list)
        
        # Pruning: Try to remove redundant elements
        # An element is redundant if removing it doesn't uncover any sum.
        # We can check this by removing it and checking if any count drops to 0.
        # To save time, we only prune if len(A_list) > 20.
        if len(A_list) > 20:
            for i in range(len(A_list)):
                if time.time() > deadline - 5.0:
                    break
                x = A_list[i]
                # Temporarily remove
                A_list.pop(i)
                A_set.remove(x)
                new_counts = compute_counts(A_list)
                if np.any(new_counts == 0):
                    # Revert
                    A_list.insert(i, x)
                    A_set.add(x)
                else:
                    # Keep removed
                    current_counts = new_counts
                    break # Restart pruning loop as indices changed
    
    # Ensure we have a valid covering set
    if np.any(compute_counts(A_list) == 0):
        # Fallback: Add random elements until covered
        while time.time() < deadline - 2.0:
            if not np.any(compute_counts(A_list) == 0):
                break
            x = random.randint(0, N)
            if x not in A_set:
                A_list.append(x)
                A_set.add(x)
    
    # --- Phase 2: Local Search ---
    
    best_A = list(A_list)
    best_cost = get_cost(best_A)
    current_A = list(A_list)
    current_set = set(A_list)
    current_cost = best_cost
    current_smooth = get_smooth_cost(current_A, beta=1.0)
    
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Precompute candidates for swaps
    all_elements = set(range(0, N + 1))
    
    iteration = 0
    max_iterations = 100000
    stuck_count = 0
    max_stuck = 2000
    
    while time.time() < deadline - 2.0:
        iteration += 1
        T = max(T * cooling_rate, min_T)
        
        r = random.random()
        
        # Move types:
        # 1. Swap (0.6)
        # 2. Remove (0.2) - only if len > 17
        # 3. Add (0.1) - only if len < 25
        # 4. Double Swap (0.1)
        
        if r < 0.6:
            # Swap
            idx = random.randint(0, len(current_A) - 1)
            x_in = current_A[idx]
            # Choose x_out not in current_set
            candidates = list(all_elements - current_set)
            if not candidates:
                continue
            x_out = random.choice(candidates)
            
            current_A[idx] = x_out
            current_set.remove(x_in)
            current_set.add(x_out)
            
            new_cost = get_cost(current_A)
            new_smooth = get_smooth_cost(current_A, beta=1.0)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(current_A)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                # Revert
                current_A[idx] = x_in
                current_set.remove(x_out)
                current_set.add(x_in)
                stuck_count += 1
                
        elif r < 0.8 and len(current_A) > 17:
            # Remove
            idx = random.randint(0, len(current_A) - 1)
            x = current_A[idx]
            current_A.pop(idx)
            current_set.remove(x)
            
            new_cost = get_cost(current_A)
            new_smooth = get_smooth_cost(current_A, beta=1.0)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(current_A)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                current_A.append(x)
                current_set.add(x)
                stuck_count += 1
                
        elif r < 0.9 and len(current_A) < 25:
            # Add
            candidates = list(all_elements - current_set)
            if not candidates:
                continue
            x = random.choice(candidates)
            current_A.append(x)
            current_set.add(x)
            
            new_cost = get_cost(current_A)
            new_smooth = get_smooth_cost(current_A, beta=1.0)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(current_A)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                current_A.remove(x)
                current_set.remove(x)
                stuck_count += 1
                
        else:
            # Double Swap
            if len(current_A) >= 2:
                idx1, idx2 = random.sample(range(len(current_A)), 2)
                x1 = current_A[idx1]
                x2 = current_A[idx2]
                
                candidates = list(all_elements - current_set)
                if len(candidates) < 2:
                    continue
                
                x_out1, x_out2 = random.sample(candidates, 2)
                
                current_A[idx1] = x_out1
                current_A[idx2] = x_out2
                current_set.remove(x1)
                current_set.remove(x2)
                current_set.add(x_out1)
                current_set.add(x_out2)
                
                new_cost = get_cost(current_A)
                new_smooth = get_smooth_cost(current_A, beta=1.0)
                
                delta = new_smooth - current_smooth
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                    current_cost = new_cost
                    current_smooth = new_smooth
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = list(current_A)
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    current_A[idx1] = x1
                    current_A[idx2] = x2
                    current_set.remove(x_out1)
                    current_set.remove(x_out2)
                    current_set.add(x1)
                    current_set.add(x2)
                    stuck_count += 1
        
        # Restart if stuck
        if stuck_count > max_stuck:
            current_A = list(best_A)
            current_set = set(current_A)
            current_cost = get_cost(current_A)
            current_smooth = get_smooth_cost(current_A, beta=1.0)
            stuck_count = 0
            T = 5.0 # Reset temperature
            
        # Early exit if optimal found
        if best_cost <= 2:
            break

    # Final Polish: Deterministic local search on best_A
    if time.time() < deadline - 1.0:
        polish_A = list(best_A)
        polish_set = set(polish_A)
        polish_cost = get_cost(polish_A)
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            
            # Try removing each element
            for i in range(len(polish_A)):
                x = polish_A[i]
                polish_A.pop(i)
                polish_set.remove(x)
                new_cost = get_cost(polish_A)
                if new_cost < polish_cost:
                    polish_cost = new_cost
                    improved = True
                    break
                else:
                    polish_A.insert(i, x)
                    polish_set.add(x)
            
            if not improved:
                # Try swapping each element with any other
                for i in range(len(polish_A)):
                    if time.time() > deadline - 0.5:
                        break
                    x_in = polish_A[i]
                    for x_out in range(N + 1):
                        if x_out in polish_set:
                            continue
                        polish_A[i] = x_out
                        polish_set.remove(x_in)
                        polish_set.add(x_out)
                        new_cost = get_cost(polish_A)
                        if new_cost < polish_cost:
                            polish_cost = new_cost
                            improved = True
                            break
                        else:
                            polish_A[i] = x_in
                            polish_set.remove(x_out)
                            polish_set.add(x_in)
                    if improved:
                        break
            
            if polish_cost < best_cost:
                best_cost = polish_cost
                best_A = list(polish_A)
    
    return list(best_A)
