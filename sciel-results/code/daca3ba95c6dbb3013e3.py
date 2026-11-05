import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    target_min = 128
    target_max = 256
    
    # Precompute indices for fast lookups
    all_indices = np.arange(129)
    
    def calculate_counts(A_mask):
        """
        Vectorized calculation of pair sum counts.
        A_mask: boolean array of length 129.
        Returns: array of counts for sums 0..256.
        """
        A_indices = np.where(A_mask)[0]
        if len(A_indices) == 0:
            return np.zeros(257, dtype=np.int32)
        
        # Compute all pair sums (i <= j)
        # This creates a matrix of sums, but we can be smarter.
        # For each a in A, add a to all elements in A.
        # Since we need unordered pairs with replacement (i<=j), 
        # we can compute the full convolution and adjust, or just use broadcasting carefully.
        
        # Method: 
        # Let A be the set of values.
        # We want count of pairs (a,b) with a<=b such that a+b=s.
        # Note: The problem says "Pair representations are unordered and permit equal summands."
        # This corresponds to the number of solutions to a+b=s with a,b in A and a<=b.
        
        # Efficient way:
        # 1. Compute all sums a+b for a,b in A.
        # 2. For a < b, the pair (a,b) is counted once in the full sum matrix (at a,b and b,a).
        #    For a = b, the pair (a,a) is counted once.
        #    So, total unordered pairs = (FullSumCount[s] + CountOfDiagonal[s]) / 2?
        #    Wait.
        #    Let N_full[s] = number of pairs (a,b) in AxA such that a+b=s.
        #    Let N_diag[s] = number of pairs (a,a) in AxA such that 2a=s.
        #    The number of unordered pairs {a,b} with a<=b is:
        #    If a != b, (a,b) and (b,a) are both in N_full. They form 1 unordered pair.
        #    If a = b, (a,a) is in N_full. It forms 1 unordered pair.
        #    So, UnorderedCount[s] = (N_full[s] - N_diag[s]) / 2 + N_diag[s] = (N_full[s] + N_diag[s]) / 2.
        
        # Let's verify:
        # A = {1, 2}. Sums: 1+1=2, 1+2=3, 2+1=3, 2+2=4.
        # N_full[2]=1, N_full[3]=2, N_full[4]=1.
        # N_diag[2]=1 (1+1), N_diag[4]=1 (2+2), N_diag[3]=0.
        # Unordered[2] = (1+1)/2 = 1. Correct ({1,1}).
        # Unordered[3] = (2+0)/2 = 1. Correct ({1,2}).
        # Unordered[4] = (1+1)/2 = 1. Correct ({2,2}).
        
        # We only care about sums in [128, 256].
        
        # To compute N_full efficiently:
        # We can use np.add.outer(A_indices, A_indices) but that's O(k^2) memory.
        # k is likely < 100. 100x100 = 10,000 ints. That's fine.
        
        sums = np.add.outer(A_indices, A_indices)
        
        # Count occurrences for each sum in the range
        # We can use np.bincount on the flattened sums, but we only need specific range.
        # Since sums are small (<= 256), bincount is fast.
        counts_full = np.bincount(sums.flatten(), minlength=257)
        
        # Calculate N_diag: 2*a = s => a = s/2.
        # For each s in target range, check if s/2 is in A and s is even.
        # Or simply: for each a in A, if 2a is in range, increment diag count.
        counts_diag = np.zeros(257, dtype=np.int32)
        for a in A_indices:
            s = 2 * a
            if 0 <= s <= 256:
                counts_diag[s] += 1
        
        # Final counts
        counts = (counts_full + counts_diag) // 2
        return counts

    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = np.max(target_N)
        sum_sq = np.sum(target_N**2)
        return (0, peak, sum_sq)

    def greedy_init():
        # Start with a set that guarantees coverage.
        # A = [0, 128] covers 128. 
        # To cover all [128, 256], we can use a "basis" approach.
        # If we take A = [0, 1, ..., k], it covers [0, 2k].
        # We need to cover [128, 256].
        # Let's try a random set of size ~40-50 and repair.
        
        A_mask = np.zeros(129, dtype=bool)
        
        # Random initialization
        k = random.randint(30, 50)
        init_set = set(random.sample(range(129), k))
        for x in init_set:
            A_mask[x] = True
            
        N = calculate_counts(A_mask)
        
        # Repair: Add elements to cover missing sums
        # Iterate until no missing sums or no improvement
        for _ in range(100):
            missing_sums = [s for s in range(target_min, target_max + 1) if N[s] == 0]
            if not missing_sums:
                break
            
            best_x = None
            best_score = -1
            best_new_N = None
            
            candidates = [x for x in range(129) if not A_mask[x]]
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            # Limit candidates checked per step for speed, but vectorized count is fast enough
            # to check all or a large subset. Let's check all for now.
            for x in candidates:
                # Quick check: Does x help?
                # x + a = s => a = s - x.
                # If there is a missing s, and s-x is in A, then adding x covers s.
                # We want to maximize number of missing sums covered.
                
                covered = 0
                for s in missing_sums:
                    a = s - x
                    if 0 <= a <= 128 and A_mask[a]:
                        covered += 1
                    # Also check if x+x = s (if x not in A yet, this is new pair)
                    # Wait, if we add x, the new pairs are (x, a) for all a in A, and (x, x).
                    # So if s - x is in A, s is covered.
                    # If 2x = s, s is covered (since x is added).
                    if 2 * x == s:
                         # This is covered by the logic above if we consider x in A? 
                         # No, x is not in A yet. 
                         # But (x,x) is a valid pair once x is in A.
                         # So yes, if 2x==s, it's covered.
                         pass # Already counted if s-x == x? i.e. s=2x. 
                         # If s=2x, then a=x. But x is not in A yet. 
                         # So the condition `A_mask[a]` would be False.
                         # We must explicitly check 2x==s.
                         # My loop `if 0 <= a <= 128 and A_mask[a]` misses the case where a=x (not in A).
                         # So we need to add: if 2*x == s, covered += 1 (but avoid double counting if s-x is in A? 
                         # If s=2x, and x not in A, then s-x=x not in A. So no double count.)
                         if 2*x == s:
                             covered += 1
                             # Wait, I added it in the loop? No, I put it inside the for s loop.
                             # Let's refine:
                             # If s-x is in A, cover.
                             # If 2x == s, cover.
                             # These are disjoint if x not in A.
                             # If x in A, we wouldn't be adding it.
                             # So just:
                             # if (0 <= s-x <= 128 and A_mask[s-x]) or (2*x == s): covered += 1
                             # But I already did `if 0 <= a <= 128 and A_mask[a]: covered += 1`.
                             # So I just need to add the 2x==s case if it wasn't covered.
                             # Since x not in A, A_mask[x] is False, so the first condition is False for a=x.
                             # So I can just add:
                             # if 2*x == s: covered += 1
                             # But I put it inside the `if 2*x == s:` block which was just `pass`.
                             # Let's fix the logic in the actual code below.
                             pass
                             
                # Re-evaluating the logic above, it's better to just compute the new N for the best candidate.
                # But that's expensive.
                # Let's stick to the heuristic: count how many missing sums s have s-x in A OR 2x==s.
                
                if covered > best_score:
                    best_score = covered
                    best_x = x
                    best_new_N = None # We will recompute N after adding
                    
            if best_x is None or best_score == 0:
                break
                
            A_mask[best_x] = True
            N = calculate_counts(A_mask)
            
        return A_mask, N

    best_peak = float('inf')
    best_A_mask = None
    best_sum_sq = float('inf')

    # Main Search Loop
    # We will run multiple restarts.
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        A_mask, N = greedy_init()
        obj = get_objective(N)
        
        if obj[0] > 0:
            # Try harder repair or skip
            continue
            
        current_peak = obj[1]
        current_sum_sq = obj[2]
        
        if current_peak < best_peak or (current_peak == best_peak and current_sum_sq < best_sum_sq):
            best_peak = current_peak
            best_sum_sq = current_sum_sq
            best_A_mask = A_mask.copy()
            
        # Local Search
        # Time limit per local search
        ls_deadline = time.time() + 3.0 # 3 seconds per restart
        
        while time.time() < ls_deadline:
            improved = False
            
            # Try swaps: remove a, add b
            # To speed up, we can sample random a and b.
            # Or iterate through all a in A and random b not in A.
            
            A_indices = np.where(A_mask)[0]
            if len(A_indices) == 0:
                break
                
            # Randomly pick a few candidates for swap
            for _ in range(50): # 50 swap attempts per iteration
                if time.time() > ls_deadline:
                    break
                    
                a = A_indices[random.randint(0, len(A_indices)-1)]
                # Pick b not in A
                # To avoid O(129) search for b, we can precompute non-members or just random guess
                # Since |A| is small, random guess is fine.
                b = random.randint(0, 128)
                if A_mask[b]:
                    continue
                if a == b:
                    continue
                    
                # Perform swap
                A_mask[a] = False
                A_mask[b] = True
                
                N_new = calculate_counts(A_mask)
                obj_new = get_objective(N_new)
                
                if obj_new[0] == 0: # Valid
                    new_peak = obj_new[1]
                    new_sq = obj_new[2]
                    
                    if (new_peak < current_peak) or (new_peak == current_peak and new_sq < current_sum_sq):
                        current_peak = new_peak
                        current_sum_sq = new_sq
                        N = N_new
                        improved = True
                        
                        if current_peak < best_peak or (current_peak == best_peak and current_sum_sq < best_sum_sq):
                            best_peak = current_peak
                            best_sum_sq = current_sum_sq
                            best_A_mask = A_mask.copy()
                    else:
                        # Revert
                        A_mask[a] = True
                        A_mask[b] = False
                else:
                    # Invalid, revert
                    A_mask[a] = True
                    A_mask[b] = False
            
            if not improved:
                # Try random perturbation to escape local minima
                # Flip 1-3 bits
                flips = random.randint(1, 3)
                for _ in range(flips):
                    x = random.randint(0, 128)
                    A_mask[x] = not A_mask[x]
                
                N = calculate_counts(A_mask)
                obj = get_objective(N)
                if obj[0] > 0:
                    # Repair or revert?
                    # Let's revert to last valid state if we saved it? 
                    # We didn't save it explicitly, but we can just continue searching from this invalid state?
                    # No, get_objective returns inf peak if invalid.
                    # Better to revert to a valid state.
                    # We don't have the previous valid state stored separately from A_mask.
                    # Let's just do a greedy repair.
                    for _ in range(20):
                        missing_sums = [s for s in range(target_min, target_max + 1) if N[s] == 0]
                        if not missing_sums:
                            break
                        # Find best x to add
                        best_x = None
                        best_score = -1
                        candidates = [x for x in range(129) if not A_mask[x]]
                        random.shuffle(candidates)
                        for x in candidates[:20]: # Limit check
                            covered = 0
                            for s in missing_sums:
                                if (0 <= s-x <= 128 and A_mask[s-x]) or (2*x == s):
                                    covered += 1
                            if covered > best_score:
                                best_score = covered
                                best_x = x
                        if best_x is None or best_score == 0:
                            break
                        A_mask[best_x] = True
                        N = calculate_counts(A_mask)
                    
                    obj = get_objective(N)
                    if obj[0] > 0:
                        # Still invalid, break local search for this restart
                        break
                    else:
                        current_peak = obj[1]
                        current_sum_sq = obj[2]

    if best_A_mask is None:
        # Fallback
        return list(range(64, 129))
        
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
