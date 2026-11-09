import time
import random

def construct():
    n = 256
    deadline = time.time() + 50  # Safety margin
    
    # Precompute all arithmetic progressions (APs)
    # We need to consider all APs contained in [0, n-1]
    # An AP is defined by start i, step k, length L
    # We will store them as lists of indices for fast summing later, 
    # but for greedy construction, we need to know which APs end at j.
    
    # For efficiency in the greedy step, we can precompute for each index j,
    # the list of APs that end at j.
    # However, the number of APs is large.
    # Let's estimate:
    # Sum_{k=1}^{n} Sum_{i=0}^{n-1} floor((n-1-i)/k) + 1
    # This is roughly O(n^2 log n).
    # For n=256, this is manageable.
    
    # Actually, a simpler way for the greedy step:
    # When we decide x_j, we want to minimize max_{AP ending at j} |sum(AP)|.
    # But sum(AP) depends on previous x's.
    # Let S_j be the current partial sums for all APs ending at j.
    # If we set x_j = 1, new sums are S_j + 1.
    # If we set x_j = -1, new sums are S_j - 1.
    # We want to choose the sign that minimizes the maximum absolute value of the new sums.
    # Let M_1 = max_{AP ending at j} |S_AP + 1|
    # Let M_2 = max_{AP ending at j} |S_AP - 1|
    # Choose 1 if M_1 < M_2, else -1 (with tie-breaking).
    
    # We also need to track the global max over ALL APs, not just those ending at j.
    # But APs not ending at j are unaffected by x_j.
    # So the global max after setting x_j is max(global_max_prev, M_choice).
    # Wait, global_max_prev is the max over all APs that do NOT end at j?
    # No, global_max_prev is the max over all APs in the prefix [0, j-1].
    # APs that end at j are the only ones whose sum changes.
    # So yes, new_global_max = max(old_global_max_excluding_ending_at_j, M_choice).
    # Actually, old_global_max might have been achieved by an AP ending at j-1 or earlier.
    # Those APs are unchanged.
    # So we just need to know the max sum of APs that do NOT end at j.
    # Let's maintain a structure that allows us to query the max sum of APs not ending at j.
    # This is hard to maintain dynamically.
    
    # Alternative:
    # Just recompute the max over all APs ending at j for both choices.
    # And keep track of the current global max.
    # Is it possible that the current global max was from an AP ending at j?
    # No, because we are constructing j. The APs ending at j are being formed now.
    # The APs that existed before (ending at < j) are fixed.
    # So, let Max_Prev be the max absolute sum of all APs contained in [0, j-1].
    # Then new_Max = max(Max_Prev, M_choice).
    # We want to minimize new_Max.
    # So we want to minimize M_choice.
    
    # So the greedy rule is:
    # 1. Compute Max_Prev (max abs sum of all APs in [0, j-1]).
    # 2. For each AP ending at j, compute the sum of the part in [0, j-1].
    # 3. For x_j=1, compute max |sum_part + 1|.
    # 4. For x_j=-1, compute max |sum_part - 1|.
    # 5. Choose x_j that gives the smaller of these two maxima.
    
    # To do this efficiently, we need to be able to compute sum_part for all APs ending at j.
    # An AP ending at j with step k is (j-(L-1)k, ..., j).
    # The part in [0, j-1] is (j-(L-1)k, ..., j-k).
    # This is just the sum of the AP starting at j-(L-1)k with step k and length L-1.
    
    # We can precompute for each j, the list of (start, step) pairs for APs ending at j.
    # But we need the sum of the prefix.
    # We can maintain an array of prefix sums for each step?
    # Let P_k[i] = sum of x_i + x_{i+k} + x_{i+2k} + ... up to some limit?
    # This is getting complicated.
    
    # Let's try a simpler, slightly less efficient but robust approach:
    # Maintain the sequence x.
    # At step j, for each possible AP ending at j (defined by start i and step k such that i + m*k = j),
    # we compute the sum of x[i...j-k] (the part before j).
    # There are O(j log j) such APs? No, O(j) for each k, sum over k is O(j log j).
    # For n=256, j=255, sum_{k=1}^{255} 255/k ~ 255 * 6 ~ 1500 operations.
    # Total operations ~ sum_{j=1}^{256} j log j ~ 256^2/2 * 6 ~ 200,000.
    # This is very fast.
    
    # So the plan:
    # 1. x = [0] * n
    # 2. Max_Prev = 0
    # 3. For j in range(n):
    #    a. For each step k in 1..j:
    #       For each start i in 0..j-k (step k):
    #          If i + m*k == j for some m>=1:
    #             Compute sum of x[i:j:k] (excluding x[j])
    #             s1 = |sum + 1|
    #             s2 = |sum - 1|
    #             track max_s1, max_s2
    #    b. M_1 = max_s1, M_2 = max_s2
    #    c. If M_1 < M_2: x[j] = 1
    #       Else: x[j] = -1
    #    d. Update Max_Prev = max(Max_Prev, min(M_1, M_2))
    #       Wait, Max_Prev should be the max over ALL APs in [0, j].
    #       The APs in [0, j] are either in [0, j-1] or end at j.
    #       So Max_Prev_new = max(Max_Prev_old, M_choice).
    
    # Let's refine the loop for speed.
    # Instead of iterating all starts for each k, we can iterate backwards from j.
    # For a fixed j and k, the APs ending at j are:
    # j, j-k, j-2k, ...
    # The sum of the part before j is sum(x[j-k], x[j-2k], ...).
    # Let S_k(j) be the sum of the AP ending at j with step k, excluding the last term.
    # S_k(j) = x[j-k] + S_k(j-k)
    # We can maintain an array S_k of size n for each k?
    # That would be n^2 memory, which is 256^2 = 65k, fine.
    # But we only need S_k(j) for the current j.
    # Actually, we can just compute it on the fly if we have the previous values.
    # Let's use a 2D array prev_sum[k][j] = sum of x[j-k] + x[j-2k] + ...
    # prev_sum[k][j] = x[j-k] + prev_sum[k][j-k]
    
    # Let's implement this.
    
    x = [0] * n
    # prev_sum[k][j] stores sum of x[j-k] + x[j-2k] + ...
    # We only need to keep track of the current column j?
    # No, to compute prev_sum[k][j], we need prev_sum[k][j-k].
    # So we need to store the whole table or at least the diagonal.
    # Since n is small, let's just store the whole table.
    # prev_sum is a list of lists, or a 2D numpy array.
    # Let's use a list of lists for simplicity and speed in Python loops?
    # Or just compute on the fly?
    # On the fly:
    # For j in range(n):
    #   max_s1 = 0
    #   max_s2 = 0
    #   for k in range(1, j+1):
    #     s = 0
    #     idx = j - k
    #     while idx >= 0:
    #       s += x[idx]
    #       idx -= k
    #     s1 = abs(s + 1)
    #     s2 = abs(s - 1)
    #     if s1 > max_s1: max_s1 = s1
    #     if s2 > max_s2: max_s2 = s2
    #   # Also consider singleton AP [j]
    #   # Sum is 0 before j.
    #   # s1 = |0+1| = 1, s2 = |0-1| = 1.
    #   # So max_s1 >= 1, max_s2 >= 1.
    #   if max_s1 < max_s2:
    #     x[j] = 1
    #     current_max = max_s1
    #   else:
    #     x[j] = -1
    #     current_max = max_s2
    #   # Update global max
    #   if current_max > global_max:
    #     global_max = current_max
    
    # This is O(n^3) in the worst case?
    # Inner loop runs j/k times.
    # Sum_{k=1}^j j/k = j * H_j ~ j log j.
    # Total sum_{j=1}^n j log j ~ n^2 log n / 2.
    # For n=256, 256^2 * 8 / 2 ~ 262,000 operations.
    # This is very fast.
    
    global_max = 0
    
    for j in range(n):
        max_s1 = 1  # Singleton
        max_s2 = 1  # Singleton
        
        for k in range(1, j + 1):
            s = 0
            idx = j - k
            while idx >= 0:
                s += x[idx]
                idx -= k
            s1 = abs(s + 1)
            s2 = abs(s - 1)
            if s1 > max_s1:
                max_s1 = s1
            if s2 > max_s2:
                max_s2 = s2
        
        if max_s1 < max_s2:
            x[j] = 1
            current_max = max_s1
        else:
            x[j] = -1
            current_max = max_s2
        
        if current_max > global_max:
            global_max = current_max
            
    # After greedy construction, we have a good starting point.
    # Now, let's do a short local search to refine.
    # The greedy choice is myopic. It doesn't consider how x[j] affects future APs.
    # Flipping a sign might reduce the max in the future.
    
    def get_cost(seq):
        max_abs = 0
        s = seq
        nn = n
        # Only check APs that could potentially have high sums?
        # No, we must check all.
        # But we can prune: if current sum + remaining terms <= max_abs, skip?
        # Hard to do efficiently.
        # Let's just do the standard check.
        for k in range(1, nn):
            for i in range(nn):
                curr = 0
                idx = i
                while idx < nn:
                    curr += s[idx]
                    if curr > max_abs:
                        max_abs = curr
                    elif curr < -max_abs:
                        max_abs = -curr
                    idx += k
        return max_abs

    best_seq = x[:]
    best_cost = get_cost(best_seq)
    
    rng = random.Random(42)
    seq = x[:]
    
    # Simple hill climbing with restarts
    no_improve = 0
    patience = 50
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.randrange(n)
        old_val = seq[idx]
        seq[idx] = -old_val
        
        new_cost = get_cost(seq)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_seq = seq[:]
            no_improve = 0
        else:
            seq[idx] = old_val
            no_improve += 1
        
        if no_improve > patience:
            # Restart with some noise from best
            seq = best_seq[:]
            for _ in range(3):
                j = rng.randrange(n)
                seq[j] *= -1
            no_improve = 0
            
    return best_seq

# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Result: {res}")
    # Verify cost
    max_abs = 0
    n = 256
    for k in range(1, n):
        for i in range(n):
            curr = 0
            idx = i
            while idx < n:
                curr += res[idx]
                if curr > max_abs:
                    max_abs = curr
                elif curr < -max_abs:
                    max_abs = -curr
                idx += k
    print(f"Max Abs Sum: {max_abs}")
