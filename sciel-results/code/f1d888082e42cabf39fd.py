import time
import math
import itertools
import random

def score_set(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def fast_score_set(U):
    """
    Optimized scoring for small sets.
    """
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        a = U[i]
        for j in range(n):
            b = U[j]
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial seed: a small set known to be decent
    best_set = [0, 1, 3, 7, 13]
    best_score = score_set(best_set)
    
    # Strategy: Base-b expansion with variable digit sets
    # We model U as a set of numbers formed by digits from a set D at positions 0..k-1
    # U = { sum d_i * b^i | d_i in D }
    # We search over base b and digit set D.
    
    # To allow more flexibility, we can also add a "shift" or "scale" but let's stick to pure expansion first.
    # The size of U is |D|^k.
    # We want to maximize score.
    
    # Phase 1: Search over small bases and digit sets
    # b from 2 to 10
    # k from 1 to 5 (since 10^5 is large, but we can limit max element)
    # D is a subset of {0, ..., b-1} containing 0.
    
    # Let's iterate over possible structures.
    # For a fixed b and k, the max element is roughly (b-1) * (b^k - 1) / (b - 1) = b^k - 1.
    # The denominator log(2*max+1) is approx k*log(b).
    # The numerator log(|U-U|/|U+U|) depends on the structure.
    # For "nice" digit sets (like {0, b-1}), the sumset and difference set have predictable sizes.
    
    # Let's generate candidates based on this parameterization.
    
    candidates = set()
    
    # Add the initial best
    candidates.add(tuple(sorted(best_set)))
    
    # Search bases 2 to 12
    for b in range(2, 13):
        # Max k such that max element < 10000 (to keep scoring fast? No, scoring is O(n^2), n=|D|^k)
        # If |D|=2, k=10 -> n=1024, n^2=1M ops. Fast.
        # If |D|=3, k=7 -> n=2187, n^2=4.7M ops. Still okay in Python? Maybe tight.
        # Let's limit n to 5000 for safety in scoring.
        
        max_n = 5000
        
        # Iterate over digit set sizes
        # D must contain 0.
        # Let m = |D|.
        # m can be 1 to b.
        
        for m in range(1, b + 1):
            # Max k such that m^k <= max_n
            max_k = 0
            prod = 1
            while prod * m <= max_n:
                prod *= m
                max_k += 1
            
            if max_k == 0:
                continue
                
            # We can't iterate all subsets of digits for large b.
            # But for small b, we can.
            # If b is large, we only try specific "structured" digit sets.
            
            if b <= 8:
                # Iterate all subsets of {0..b-1} of size m containing 0
                other_digits = list(range(1, b))
                for D_other in itertools.combinations(other_digits, m - 1):
                    D = [0] + list(D_other)
                    D.sort()
                    
                    # For each valid k
                    for k in range(1, max_k + 1):
                        # Construct U
                        U = set()
                        # Efficient construction
                        # Start with {0}
                        current_layer = {0}
                        for i in range(k):
                            next_layer = set()
                            for val in current_layer:
                                for d in D:
                                    next_layer.add(val + d * (b ** i))
                            current_layer = next_layer
                        U = list(current_layer)
                        
                        if not U or 0 not in U:
                            continue
                        if max(U) > 100000: # Extra safety on max value
                            continue
                            
                        s = tuple(sorted(U))
                        candidates.add(s)
                        
                        if time.time() > deadline:
                            return list(best_set)
            else:
                # For larger bases, try only "extreme" digit sets
                # e.g., {0, b-1}, {0, 1, b-1}, etc.
                extreme_sets = []
                # {0, b-1}
                if m >= 2:
                    extreme_sets.append([0, b-1])
                # {0, 1, b-1}
                if m >= 3:
                    extreme_sets.append([0, 1, b-1])
                # {0, b//2, b-1}
                if m >= 3 and b % 2 == 0:
                    extreme_sets.append([0, b//2, b-1])
                
                for D in extreme_sets:
                    if len(D) != m:
                        # If we can't match exact size m with these simple sets, skip or adjust
                        # Actually, let's just use the sets we generated, regardless of target m, 
                        # but ensure they are distinct and valid.
                        # We will just evaluate them.
                        pass
                    
                    D.sort()
                    # Ensure 0 is in D
                    if 0 not in D:
                        continue
                        
                    for k in range(1, max_k + 1):
                        U = set()
                        current_layer = {0}
                        for i in range(k):
                            next_layer = set()
                            for val in current_layer:
                                for d in D:
                                    next_layer.add(val + d * (b ** i))
                            current_layer = next_layer
                        U = list(current_layer)
                        
                        if not U or 0 not in U:
                            continue
                        if max(U) > 100000:
                            continue
                            
                        s = tuple(sorted(U))
                        candidates.add(s)
                        
                        if time.time() > deadline:
                            return list(best_set)

    # Phase 2: Evaluate all candidates
    # This might be many, so we filter.
    # Actually, let's just evaluate the top few most promising ones?
    # Or just evaluate all if count is reasonable.
    
    # To be safe, limit evaluation count if it's huge
    cand_list = list(candidates)
    if len(cand_list) > 2000:
        # Randomly sample? Or just take first 2000?
        # Let's take a random sample to be unbiased
        random.shuffle(cand_list)
        cand_list = cand_list[:2000]
        
    for U in cand_list:
        if time.time() > deadline:
            break
        sc = fast_score_set(U)
        if sc > best_score:
            best_score = sc
            best_set = list(U)

    # Phase 3: Local Search on the best found set
    # Perturb elements slightly to break the rigid structure if it helps.
    current = best_set[:]
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        max_search_val = max(current) + 10
        if max_search_val > 200:
            max_search_val = 200
            
        # Try swapping each element
        for i in range(len(current)):
            old_val = current[i]
            for new_val in range(max(0, old_val - 5), min(max_search_val, old_val + 5) + 1):
                if new_val == old_val:
                    continue
                if old_val == 0 and new_val != 0:
                    continue
                if new_val < 0:
                    continue
                    
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if 0 not in new_set:
                    continue
                if len(new_set) < 2:
                    continue
                
                s = fast_score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                        
        # Try adding an element
        if len(current) < 50: # Cap size for local search speed
            for new_val in range(0, max_search_val + 1, 2): # Step 2 for speed
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                if 0 not in new_set:
                    continue
                s = fast_score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                        
        # Try removing an element
        if len(current) > 2:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                new_set = current[:i] + current[i+1:]
                s = fast_score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]

    return best_set
