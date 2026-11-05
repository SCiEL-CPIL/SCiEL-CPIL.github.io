import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    
    # We will search for sets of specific sizes
    # Based on analysis, k=16 is the minimum for M=1 potentially.
    # k=18-22 is safer for finding low M.
    k_candidates = [16, 18, 20, 22, 24]
    
    best_A = None
    best_score = float('inf')
    
    # Helper to initialize counts for a set A
    def init_counts(A):
        counts = np.zeros(max_sum + 1, dtype=int)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            # Self pair
            counts[2*a] += 1
            # Pairs with larger indices to avoid double counting
            for j in range(i+1, len(A_list)):
                b = A_list[j]
                counts[a+b] += 1
        return counts

    # Helper to get cost from counts
    def get_cost_from_counts(counts):
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return 10000 + int(np.sum(target_counts == 0))
        return int(np.max(target_counts))

    def local_search(A_init, k, time_limit):
        A = set(A_init)
        # Ensure size is exactly k? 
        # If len(A) != k, we should adjust first.
        # For simplicity, let's stick to fixed k in the search.
        if len(A) != k:
            # Resize
            if len(A) > k:
                # Remove random elements
                to_remove = random.sample(list(A), len(A) - k)
                for x in to_remove:
                    A.remove(x)
            else:
                # Add random elements
                while len(A) < k:
                    x = random.randint(0, N)
                    if x not in A:
                        A.add(x)
        
        counts = init_counts(A)
        current_cost = get_cost_from_counts(counts)
        best_local_cost = current_cost
        best_local_A = set(A)
        
        all_elements = set(range(N + 1))
        
        # Precompute list of elements in A for faster iteration
        A_list = list(A)
        
        while time.time() < time_limit:
            # Check for optimal
            if best_local_cost <= 1:
                break
                
            # Determine strategy based on current state
            target_counts = counts[min_sum:max_sum+1]
            uncovered = np.where(target_counts == 0)[0]
            
            if len(uncovered) > 0:
                # Priority: Fix coverage
                # Find a swap that covers the most uncovered sums
                best_gain = -1
                best_swap = None
                
                # Sample some swaps to speed up
                A_list = list(A)
                random.shuffle(A_list)
                
                for x_out in A_list[:10]: # Try removing one of these
                    candidates = list(all_elements - A)
                    random.shuffle(candidates)
                    
                    for x_in in candidates[:10]: # Try adding one of these
                        # Calculate gain: how many uncovered sums does x_in cover that x_out didn't?
                        # Actually, we need to see if the swap reduces the number of uncovered sums.
                        # A sum s is covered if count[s] > 0.
                        # Removing x_out decreases count[x_out+y] for y in A.
                        # Adding x_in increases count[x_in+y] for y in A.
                        
                        # This incremental check is complex. 
                        # Let's just do the swap, check cost, and revert if bad.
                        # Since k is small, this is fast.
                        
                        old_counts = counts.copy()
                        
                        # Remove x_out
                        for y in A:
                            if y == x_out:
                                continue
                            s = x_out + y
                            counts[s] -= 1
                        counts[2*x_out] -= 1
                        
                        # Add x_in
                        for y in A:
                            if y == x_out: # y is still in A at this point in logic? 
                                # A is not modified yet, just counts.
                                # We are simulating.
                                pass
                            s = x_in + y
                            counts[s] += 1
                        counts[2*x_in] += 1
                        
                        new_cost = get_cost_from_counts(counts)
                        
                        if new_cost < current_cost:
                            # Accept
                            A.remove(x_out)
                            A.add(x_in)
                            current_cost = new_cost
                            if current_cost < best_local_cost:
                                best_local_cost = current_cost
                                best_local_A = set(A)
                            break
                        else:
                            # Revert
                            counts = old_counts
                        
                    if current_cost < best_local_cost:
                        break
                else:
                    # No improvement found in this batch, try random perturbation
                    if random.random() < 0.2:
                        x_out = random.choice(list(A))
                        x_in = random.choice(list(all_elements - A))
                        A.remove(x_out)
                        A.add(x_in)
                        counts = init_counts(A)
                        current_cost = get_cost_from_counts(counts)
                        
            else:
                # Coverage is complete. Minimize peak.
                peak = current_cost
                if peak <= 1:
                    break
                
                # Find sums with max count
                # target_counts is already extracted
                max_val = np.max(target_counts)
                peak_sums = np.where(target_counts == max_val)[0] + min_sum
                
                # Pick a random peak sum
                s_peak = random.choice(peak_sums)
                
                # Find pairs in A that sum to s_peak
                pairs = []
                A_list = list(A)
                for i in range(len(A_list)):
                    a = A_list[i]
                    b = s_peak - a
                    if b in A and a <= b: # Unordered
                        pairs.append((a, b))
                
                if not pairs:
                    # Should not happen if count > 0, but safety
                    continue
                
                # Try to break one of these pairs by swapping one of its elements
                improved = False
                for a, b in pairs:
                    # Try swapping a
                    candidates = list(all_elements - A)
                    random.shuffle(candidates)
                    
                    for x_in in candidates[:5]:
                        old_counts = counts.copy()
                        
                        # Swap a -> x_in
                        # Remove a
                        for y in A:
                            if y == a:
                                continue
                            s = a + y
                            counts[s] -= 1
                        counts[2*a] -= 1
                        
                        # Add x_in
                        for y in A:
                            if y == a:
                                continue
                            s = x_in + y
                            counts[s] += 1
                        counts[2*x_in] += 1
                        
                        new_cost = get_cost_from_counts(counts)
                        
                        if new_cost < current_cost:
                            A.remove(a)
                            A.add(x_in)
                            current_cost = new_cost
                            improved = True
                            if current_cost < best_local_cost:
                                best_local_cost = current_cost
                                best_local_A = set(A)
                            break
                        else:
                            counts = old_counts
                        
                    if improved:
                        break
                    
                    # Try swapping b
                    if not improved:
                        candidates = list(all_elements - A)
                        random.shuffle(candidates)
                        for x_in in candidates[:5]:
                            old_counts = counts.copy()
                            
                            # Swap b -> x_in
                            for y in A:
                                if y == b:
                                    continue
                                s = b + y
                                counts[s] -= 1
                            counts[2*b] -= 1
                            
                            for y in A:
                                if y == b:
                                    continue
                                s = x_in + y
                                counts[s] += 1
                            counts[2*x_in] += 1
                            
                            new_cost = get_cost_from_counts(counts)
                            
                            if new_cost < current_cost:
                                A.remove(b)
                                A.add(x_in)
                                current_cost = new_cost
                                improved = True
                                if current_cost < best_local_cost:
                                    best_local_cost = current_cost
                                    best_local_A = set(A)
                                break
                            else:
                                counts = old_counts
                            
                        if improved:
                            break
                
                if not improved:
                    # Random kick
                    if random.random() < 0.5:
                        x_out = random.choice(list(A))
                        x_in = random.choice(list(all_elements - A))
                        A.remove(x_out)
                        A.add(x_in)
                        counts = init_counts(A)
                        current_cost = get_cost_from_counts(counts)

        return best_local_A, best_local_cost

    # Main Search Loop
    random.seed(42)
    np.random.seed(42)
    
    # Allocate time to different k values
    # We expect k=18-22 to be sweet spot.
    # Let's run multiple restarts.
    
    num_restarts = 20
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Vary k slightly
        k = random.choice(k_candidates)
        
        # Random initialization
        A_init = set(random.sample(range(N + 1), k))
        
        # Run local search
        A_res, cost_res = local_search(A_init, k, time.time() + time_per_restart)
        
        if cost_res < best_score:
            best_score = cost_res
            best_A = A_res
            
        # If we found M=1, we can stop
        if best_score <= 1:
            break

    # Fallback if nothing found (should not happen)
    if best_A is None:
        best_A = set(random.sample(range(N + 1), 20))
        
    return list(best_A)

if __name__ == "__main__":
    A = construct()
    print(f"Found set of size {len(A)}")
    # Verify
    counts = np.zeros(257, dtype=int)
    for i, a in enumerate(A):
        for b in A[i:]:
            counts[a+b] += 1
    target_counts = counts[128:257]
    if np.any(target_counts == 0):
        print("Invalid: Uncovered sums found")
        print(np.where(target_counts == 0)[0] + 128)
    else:
        print(f"Valid. Peak multiplicity: {np.max(target_counts)}")
