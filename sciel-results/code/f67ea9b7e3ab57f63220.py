import math
import time

def construct():
        divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
        n = len(divisors)
        max_val = 1000000
        lcm_val = 30030
        
        start_time = time.time()
        deadline = 80.0
        
        # Sort divisors? No, order matters for output.
        
        for W in range(1, 100):
            if time.time() > deadline:
                break
                
            # Iterate L
            # We can optimize by checking L in a smarter way?
            # For now, linear scan.
            
            for L in range(1, lcm_val + 1):
                if time.time() > deadline:
                    break
                
                # Determine candidates
                # candidates[i] is a list of possible values for divisor i
                # We want to pick one from each list, all distinct.
                
                # Optimization: If W is small, lists are small.
                # We can use a bitmask for used values if values are small?
                # Values are up to 1000000, so no.
                
                # Use a set for used values.
                # Backtracking.
                
                # To speed up, check fixed candidates first.
                fixed_vals = [None] * n
                var_indices = []
                var_cands = [None] * n
                valid = True
                
                for i, d in enumerate(divisors):
                    k_min = (L + d - 1) // d
                    k_max = (L + W) // d
                    
                    if k_min > k_max:
                        valid = False
                        break
                    
                    if k_min == k_max:
                        val = k_min * d
                        if val > max_val:
                            valid = False
                            break
                        fixed_vals[i] = val
                    else:
                        # Build list
                        cands = []
                        for k in range(k_min, k_max + 1):
                            val = k * d
                            if val > max_val:
                                break
                            cands.append(val)
                        if not cands:
                            valid = False
                            break
                        var_cands[i] = cands
                        var_indices.append(i)
                
                if not valid:
                    continue
                
                # Check fixed values for duplicates
                used = set()
                for i in range(n):
                    if fixed_vals[i] is not None:
                        if fixed_vals[i] in used:
                            valid = False
                            break
                        used.add(fixed_vals[i])
                
                if not valid:
                    continue
                
                # Solve for variables
                # Sort var_indices by number of candidates (least constraining first)
                var_indices.sort(key=lambda i: len(var_cands[i]))
                
                result = [None] * n
                # Pre-fill fixed
                for i in range(n):
                    if fixed_vals[i] is not None:
                        result[i] = fixed_vals[i]
                
                # Backtracking on var_indices
                def bt(idx):
                    if idx == len(var_indices):
                        return True
                    i = var_indices[idx]
                    for val in var_cands[i]:
                        if val not in used:
                            used.add(val)
                            result[i] = val
                            if bt(idx + 1):
                                return True
                            used.remove(val)
                            result[i] = None
                    return False
                
                if bt(0):
                    # Verify
                    if len(set(result)) == n and all(1 <= v <= max_val for v in result):
                        return result
                        
        return divisors[:]
