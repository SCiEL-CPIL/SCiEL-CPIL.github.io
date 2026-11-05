import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (evaluator kills at 90s)
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_len = max_sum - min_sum + 1  # 129
    
    def compute_counts(A_list):
        """
        Compute counts of sums for target range [128, 256] using convolution.
        Returns a numpy array of length 129 where index i corresponds to sum 128+i.
        """
        if not A_list:
            return np.zeros(target_len, dtype=np.int32)
        
        # Create indicator vector
        indicator = np.zeros(N + 1, dtype=np.int32)
        for a in A_list:
            indicator[a] = 1
        
        # Convolution gives ordered pair counts
        conv = np.convolve(indicator, indicator, mode='full')
        
        # Extract target range
        # Sum s corresponds to index s in conv (since indicator is 0-indexed)
        # We need sums from 128 to 256
        f = conv[min_sum:max_sum + 1].astype(np.int32)
        
        # Calculate g(s) = number of a in A such that 2a = s
        g = np.zeros(target_len, dtype=np.int32)
        for a in A_list:
            s = 2 * a
            if min_sum <= s <= max_sum:
                g[s - min_sum] = 1  # Since A is a set, at most 1
        
        # Unordered count = (f + g) // 2
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
    
    def get_smooth_cost(A_list, beta=2.0):
        """
        Smooth surrogate for max multiplicity using log-sum-exp.
        """
        counts = compute_counts(A_list).astype(np.float64)
        if np.any(counts == 0):
            return 10000.0 + np.sum(counts == 0)
        
        max_c = np.max(counts)
        if max_c == 0:
            return 0.0
        lse = max_c + (1.0 / beta) * np.log(np.sum(np.exp(beta * (counts - max_c))))
        return lse
    
    # --- Phase 1: Greedy Construction ---
    
    random.seed(42)
    A_list = []
    A_set = set()
    current_counts = np.zeros(target_len, dtype=np.int32)
    
    # Seed with structured elements
    seed_elements = [0, 64, 128, 32, 96]
    for x in seed_elements:
        A_list.append(x)
        A_set.add(x)
    
    current_counts = compute_counts(A_list)
    
    # Greedy loop: Add elements until all sums are covered
    while time.time() < deadline - 10.0:
        if not np.any(current_counts == 0):
            break
        
        uncovered_sums = np.where(current_counts == 0)[0]
        
        best_x = None
        best_gain = -1
        
        for x in range(N + 1):
            if x in A_set:
                continue
            
            # Calculate gain: number of uncovered sums that become covered
            gain = 0
            for y in A_list:
                s = x + y
                if min_sum <= s <= max_sum:
                    idx = s - min_sum
                    if current_counts[idx] == 0:
                        gain += 1
            s_double = 2 * x
            if min_sum <= s_double <= max_sum:
                idx = s_double - min_sum
                if current_counts[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain == 0:
            break
        
        A_list.append(best_x)
        A_set.add(best_x)
        current_counts = compute_counts(A_list)
        
        # Pruning: Try to remove redundant elements
        if len(A_list) > 15:
            for i in range(len(A_list)):
                if time.time() > deadline - 10.0:
                    break
                x = A_list[i]
                A_list.pop(i)
                A_set.remove(x)
                new_counts = compute_counts(A_list)
                if np.any(new_counts == 0):
                    A_list.insert(i, x)
                    A_set.add(x)
                else:
                    current_counts = new_counts
                    break  # Restart pruning
    
    # Ensure validity
    if np.any(compute_counts(A_list) == 0):
        while time.time() < deadline - 5.0:
            if not np.any(compute_counts(A_list) == 0):
                break
            x = random.randint(0, N)
            if x not in A_set:
                A_list.append(x)
                A_set.add(x)
    
    # --- Phase 2: Simulated Annealing ---
    
    best_A = list(A_list)
    best_cost = get_cost(best_A)
    current_A = list(A_list)
    current_set = set(A_list)
    current_smooth = get_smooth_cost(current_A, beta=2.0)
    
    T = 5.0
    cooling_rate = 0.998
    min_T = 0.1
    all_elements = set(range(0, N + 1))
    
    stuck_count = 0
    max_stuck = 1500
    
    while time.time() < deadline - 3.0:
        T = max(T * cooling_rate, min_T)
        r = random.random()
        
        if r < 0.55:
            # Swap
            if not current_A:
                continue
            idx = random.randint(0, len(current_A) - 1)
            x_in = current_A[idx]
            candidates = list(all_elements - current_set)
            if not candidates:
                continue
            x_out = random.choice(candidates)
            
            current_A[idx] = x_out
            current_set.remove(x_in)
            current_set.add(x_out)
            
            new_smooth = get_smooth_cost(current_A, beta=2.0)
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_smooth = new_smooth
                new_cost = get_cost(current_A)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = list(current_A)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                current_A[idx] = x_in
                current_set.remove(x_out)
                current_set.add(x_in)
                stuck_count += 1
                
        elif r < 0.75 and len(current_A) > 14:
            # Remove
            idx = random.randint(0, len(current_A) - 1)
            x = current_A[idx]
            current_A.pop(idx)
            current_set.remove(x)
            
            new_smooth = get_smooth_cost(current_A, beta=2.0)
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_smooth = new_smooth
                new_cost = get_cost(current_A)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = list(current_A)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                current_A.append(x)
                current_set.add(x)
                stuck_count += 1
                
        elif r < 0.85 and len(current_A) < 28:
            # Add
            candidates = list(all_elements - current_set)
            if not candidates:
                continue
            x = random.choice(candidates)
            current_A.append(x)
            current_set.add(x)
            
            new_smooth = get_smooth_cost(current_A, beta=2.0)
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_smooth = new_smooth
                new_cost = get_cost(current_A)
                if new_cost < best_cost:
                    best_cost = new_cost
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
                
                new_smooth = get_smooth_cost(current_A, beta=2.0)
                delta = new_smooth - current_smooth
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                    current_smooth = new_smooth
                    new_cost = get_cost(current_A)
                    if new_cost < best_cost:
                        best_cost = new_cost
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
            current_smooth = get_smooth_cost(current_A, beta=2.0)
            stuck_count = 0
            T = 5.0
        
        # Early exit if optimal found
        if best_cost <= 2:
            break
    
    # --- Phase 3: Deterministic Polish ---
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
    
    # Final validation
    if not best_A:
        best_A = [0, 64, 128]
    
    return list(best_A)
