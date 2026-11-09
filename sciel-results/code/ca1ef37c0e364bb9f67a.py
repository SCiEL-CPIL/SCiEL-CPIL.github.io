# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = 30
    # LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    # weights[i] = LCM / (i+1) for i in 0..29
    weights = [lcm_val // (i + 1) for i in range(n)]
    
    # Meet in the middle: split into two halves of 15
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    def gen_sums_with_paths(ws):
        """
        Generate all possible sums and store the path (coefficients) for each sum.
        Since there are 3^15 = 14,348,907 sums, storing paths is memory heavy but necessary
        to reconstruct the final coefficients.
        However, we can store just the index in the ternary expansion?
        Or better: store a dictionary mapping sum -> path?
        There are at most 14M unique sums? No, sums can repeat.
        But we only need ONE path for each sum value to find a valid combination.
        So: sum_value -> path (list of 15 coeffs)
        """
        sums_dict = {0: [0] * len(ws)}
        for i, w in enumerate(ws):
            new_sums_dict = {}
            for s, path in sums_dict.items():
                # Option 1: -w
                new_s = s - w
                new_path = path[:]
                new_path[i] = -1
                if new_s not in new_sums_dict:
                    new_sums_dict[new_s] = new_path
                    
                # Option 2: 0
                # s stays s, path stays same (but we need a copy? No, path is immutable in this context if we don't mutate)
                # But we are creating new_path for -w and +w. For 0, we can reuse the old path?
                # Actually, if s is in new_sums_dict, we keep the first one found.
                # We don't need to store multiple paths for the same sum.
                
                # Option 3: +w
                new_s2 = s + w
                new_path2 = path[:]
                new_path2[i] = 1
                if new_s2 not in new_sums_dict:
                    new_sums_dict[new_s2] = new_path2
                    
                # Option 0:
                if s not in new_sums_dict:
                    # path is already correct for this index (0)
                    new_sums_dict[s] = path[:]
                    
            sums_dict = new_sums_dict
            # Check time
            if time.time() > deadline - 5:
                 return None
        return sums_dict

    # Generate sums for first half
    sums1_dict = gen_sums_with_paths(w1)
    if sums1_dict is None:
        # Fallback: heuristic
        return [0] * (n - 2) + [1, -1]
        
    # Generate sums for second half
    sums2_dict = gen_sums_with_paths(w2)
    if sums2_dict is None:
        return [0] * (n - 2) + [1, -1]

    # Now we have two dictionaries:
    # sums1_dict: sum_val -> path1
    # sums2_dict: sum_val -> path2
    # We want min |s1 + s2| != 0
    
    # Convert keys to sorted lists for two-pointer
    keys1 = sorted(sums1_dict.keys())
    keys2 = sorted(sums2_dict.keys())
    
    # Two-pointer to find min |s1 + s2| != 0
    i = 0
    j = len(keys2) - 1
    min_abs = float('inf')
    best_s1 = None
    best_s2 = None
    n1 = len(keys1)
    n2 = len(keys2)
    
    while i < n1 and j >= 0:
        s1 = keys1[i]
        s2 = keys2[j]
        val = s1 + s2
        
        if val == 0:
            # We need the smallest non-zero.
            # Check neighbors
            # (i, j-1)
            if j > 0:
                v = s1 + keys2[j-1]
                if v != 0:
                    av = abs(v)
                    if av < min_abs:
                        min_abs = av
                        best_s1 = s1
                        best_s2 = keys2[j-1]
                        if min_abs == 1:
                            break
            # (i+1, j)
            if i < n1 - 1:
                v = keys1[i+1] + s2
                if v != 0:
                    av = abs(v)
                    if av < min_abs:
                        min_abs = av
                        best_s1 = keys1[i+1]
                        best_s2 = s2
                        if min_abs == 1:
                            break
            # Move pointers
            i += 1
            j -= 1
        elif val > 0:
            av = val
            if av < min_abs:
                min_abs = av
                best_s1 = s1
                best_s2 = s2
                if min_abs == 1:
                    break
            j -= 1
        else: # val < 0
            av = -val
            if av < min_abs:
                min_abs = av
                best_s1 = s1
                best_s2 = s2
                if min_abs == 1:
                    break
            i += 1
            
    # Reconstruct coefficients
    if best_s1 is not None and best_s2 is not None:
        path1 = sums1_dict[best_s1]
        path2 = sums2_dict[best_s2]
        coeffs = path1 + path2
        return coeffs
    else:
        # Fallback
        return [0] * (n - 2) + [1, -1]

# EVOLVE-BLOCK-END
