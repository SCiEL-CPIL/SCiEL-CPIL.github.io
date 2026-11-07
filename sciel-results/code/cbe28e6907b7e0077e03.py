import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # Precompute comparability lists
    # comparables[s] is a set of masks t such that s is a strict subset of t or t is a strict subset of s
    comparables = [set() for _ in range(total_masks)]
    for s in range(total_masks):
        # s is subset of t if (s & ~t) == 0 and s != t
        # t is subset of s if (t & ~s) == 0 and t != s
        # Iterate all t? 4096 * 4096 = 16M ops. Might be slow in Python.
        # Better: iterate subsets/supersets?
        # Or just do it once at start. 16M simple bit ops is ~1-2 seconds in Python?
        # Let's try to optimize this or do it smarter.
        
    # Actually, we can compute comparables on the fly or use a faster precomputation.
    # Let's try a faster precomputation.
    # For each s, find all t such that s < t or t < s.
    # This is equivalent to: t is not equal to s, and (s & ~t) == 0 OR (t & ~s) == 0.
    
    # Let's just do the O(N^2) precomputation, it's a one-time cost.
    # 4096 * 4096 = 16,777,216.
    # Python loop might take ~5-10 seconds. That's risky.
    # Alternative: Use NumPy? Or optimize the loop.
    # Or, calculate comparability on the fly in the local search.
    # In local search, we check if x ~ B.
            # We can precompute a bitset or just use the condition.
            # Condition: (x & ~B) == 0 or (B & ~x) == 0.
            # This is very fast.
            # So we don't need to precompute `comparables` sets. We can just check the condition.
            # The cost is 2 bitwise ops and 2 comparisons.
        
            # Let's stick to the on-the-fly check but optimize the local search loop.
        
            family = []
            fam_set = set()
        
            # Start with empty set
            family.append(0)
            fam_set.add(0)
        
            # Greedy initialization
            # scores[s] = number of sets in family comparable to s
    scores = [0] * total_masks
    
    # Instead of updating all scores, we can just calculate the score of a candidate
    # by iterating over the current family.
    # Family size is small (max 40).
    # So for each candidate s, we compute:
    # score(s) = sum(1 for x in family if (s & ~x) == 0 or (x & ~s) == 0)
    
    while len(family) < m and time.time() < deadline:
        best_score = -1
        best_mask = -1
        
        # Iterate all masks
        for s in range(total_masks):
            if s in fam_set:
                continue
            
            # Calculate comparable count
            comp_count = 0
            # Unroll or use sum with generator?
            # sum(1 for x in family if (s & ~x) == 0 or (x & ~s) == 0)
            # This is O(m) per s. Total O(N * m) = 4096 * 40 = 160k ops.
            # This is fast.
            
            # Optimization: Precompute nothing, just compute.
            for x in family:
                if (s & ~x) == 0 or (x & ~s) == 0:
                    comp_count += 1
            
            if comp_count > best_score:
                best_score = comp_count
                best_mask = s
                
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        
    # Local Search
    # We will perform random restarts or just iterate until deadline.
    # Since greedy is deterministic, we might be stuck in a local optimum.
    # Let's add some randomness or try multiple greedy starts?
    # Or just do a fast local search on the result.
    
    best_family = family[:]
    best_score_val = 0
    
    # Calculate initial score
    # Score = number of comparable pairs
    # Sum of degrees / 2
    # Degree of s = count of x in family comparable to s
    total_deg = 0
    for s in family:
        deg = 0
        for x in family:
            if s != x and ((s & ~x) == 0 or (x & ~s) == 0):
                deg += 1
        total_deg += deg
    best_score_val = total_deg // 2
    
    # Local Search Loop
    # We try to swap elements to improve the score.
    # To allow escape from local optima, we can use a simple annealing or just random perturbations.
    # Given the time limit, a simple hill-climbing with random restarts or perturbations is good.
    
    current_family = family[:]
    current_score = best_score_val
    
    # Helper to calculate score of a family
    def calc_score(fam):
        total_deg = 0
        for s in fam:
            deg = 0
            for x in fam:
                if s != x and ((s & ~x) == 0 or (x & ~s) == 0):
                    deg += 1
            total_deg += deg
        return total_deg // 2

    # Fast swap evaluation
    # If we replace A with B in family F:
    # Old deg of A: dA
    # New deg of B: dB' = count of x in F\{A} comparable to B
    # Change in total deg = dB' - dA
    # Change in score = (dB' - dA) / 2
    
    # We iterate through positions and try to find a better B.
    # To speed up, we can precompute dA for all A in current_family.
    
    while time.time() < deadline:
        # Calculate current degrees
        degs = {}
        for i, A in enumerate(current_family):
            d = 0
            for j, B in enumerate(current_family):
                if i != j and ((A & ~B) == 0 or (B & ~A) == 0):
                    d += 1
            degs[A] = d # Note: if duplicates existed, this would be wrong, but they are distinct
            
        improved = False
        # Randomize the order of indices to avoid bias?
        indices = list(range(len(current_family)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            A = current_family[i]
            dA = degs[A]
            
            best_gain = 0
            best_B = -1
            
            # Try all candidates B not in current_family
            # We can iterate all masks and skip those in current_family
            # Or maintain a list of available masks?
            # Iterating 4096 is fine.
            
            for B in range(total_masks):
                if B in fam_set: # fam_set must be kept in sync!
                    continue
                
                # Calculate dB'
                # dB' = count of x in current_family \ {A} comparable to B
                # We can compute this by iterating current_family
                # Optimization: 
                # dB' = (count of x in current_family comparable to B) - (1 if A comparable to B else 0)
                
                count_B = 0
                A_comp_B = False
                for x in current_family:
                    if x == A:
                        # Check if A and B are comparable
                        if (A & ~B) == 0 or (B & ~A) == 0:
                            A_comp_B = True
                    else:
                        if (B & ~x) == 0 or (x & ~B) == 0:
                            count_B += 1
                
                # Actually, the loop above does both.
                # Let's refine:
                # total_comp_B_in_old = count of x in current_family such that x ~ B
                # dB' = total_comp_B_in_old - (1 if A ~ B else 0)
                
                # We calculated count_B as total_comp_B_in_old - (1 if A ~ B else 0) effectively?
                # No, my loop above:
                # if x == A: check comp, set flag.
                # else: check comp, increment count_B.
                # So count_B is exactly dB'.
                
                gain = count_B - dA
                if gain > best_gain:
                    best_gain = gain
                    best_B = B
            
            if best_B != -1 and best_gain > 0:
                # Perform swap
                # Update current_family
                current_family[i] = best_B
                
                # Update fam_set
                fam_set.remove(A)
                fam_set.add(best_B)
                
                # Update current_score
                current_score += best_gain // 2
                # Note: gain is change in degree sum. Score is degree sum / 2.
                # If gain is odd, score doesn't change integer-wise?
                # Degree sum is always even. So gain must be even.
                # Let's verify.
                # Removing A removes dA edges.
                # Adding B adds dB' edges.
                # Change in sum = dB' - dA.
                # Is dB' - dA always even?
                # Total degree sum is 2 * Pairs. So it's even.
                # Old sum S. New sum S' = S - dA + dB'.
                # S is even, S' is even. So dB' - dA is even.
                
                if current_score > best_score_val:
                    best_score_val = current_score
                    best_family = current_family[:]
                
                improved = True
                # Restart the loop over indices? Or continue?
                # Continuing is faster, but might miss improvements caused by the new B.
                # Since we shuffled, we'll hit other indices soon.
                
        if not improved:
            # Local optimum reached.
            # Try a random perturbation to escape?
            # Swap two random elements with random new elements?
            # Or just break if we've been stuck for a while?
            # Let's try a random restart of the local search phase with a perturbed family.
            # Or just stop if time is running out.
            # To escape, we can do a "kick": replace 2-3 random elements with random others, accept even if score drops slightly?
            # Given the time limit, simple hill climbing might be enough if the greedy start is good.
            # But greedy is deterministic.
            # Let's add a simple simulated annealing step or random kick.
            
            # Kick: Pick 2 random indices, replace with 2 random masks not in family.
            if len(current_family) >= 2:
                idx1, idx2 = random.sample(range(len(current_family)), 2)
                A1 = current_family[idx1]
                A2 = current_family[idx2]
                
                # Find random B1, B2
                # Just pick random masks
                while True:
                    B1 = random.randint(0, total_masks - 1)
                    if B1 not in fam_set:
                        break
                while True:
                    B2 = random.randint(0, total_masks - 1)
                    if B2 not in fam_set and B2 != B1:
                        break
                        
                # Apply kick
                fam_set.remove(A1)
                fam_set.remove(A2)
                fam_set.add(B1)
                fam_set.add(B2)
                
                current_family[idx1] = B1
                current_family[idx2] = B2
                
                # Recalculate score
                current_score = calc_score(current_family)
                
                if current_score > best_score_val:
                    best_score_val = current_score
                    best_family = current_family[:]

    # Restore best family
    family = best_family
    
    return _convert(family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
