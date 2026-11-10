import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    LCM = 30030
    
    start_time = time.time()
    timeout = 85
    
    best_result = None
    best_W = 1000000 # Large initial value
    
    # Sort divisors by descending value to anchor on the most constrained one
    # But we need to return in original order.
    # Let's just pick the index of the max divisor for anchoring.
    max_d_idx = max(range(n), key=lambda i: divisors[i])
    d_anchor = divisors[max_d_idx]
    
    # Iterate W from 9 upwards
    # Max W? The problem says <= 1000000. But we expect small W.
    # Let's go up to 50 or so. If not found, fallback to current method or larger W.
    # But the current method finds W=33. So W < 33.
    for W in range(9, 34):
        if time.time() - start_time > timeout:
            break
            
        # For this W, iterate over possible anchor multiples
        # The anchor value m must be a multiple of d_anchor.
        # L must be such that m is in [L, L+W].
        # So L is in [m-W, m].
        # Also L must be in [1, LCM] to cover all residues.
        # m can be in [1, LCM+W] to cover all L in [1, LCM].
        
        # To avoid duplicates, we can just iterate m from d_anchor to LCM+W
        # But we can optimize by iterating L directly? No, anchoring is better.
        
        # Let's iterate m
        # m is a multiple of d_anchor.
        # m_min = d_anchor
        # m_max = LCM + W
        # Number of multiples is (LCM+W)//d_anchor.
        
        found = False
        # To ensure we don't miss any L, we iterate m.
        # For each m, L can be m-W, ..., m.
        # We need to check if L is in [1, LCM].
        
        for m in range(d_anchor, LCM + W + 1, d_anchor):
            if time.time() - start_time > timeout:
                break
                
            # L range for this m
            L_start = max(1, m - W)
            L_end = min(LCM, m)
            
            for L in range(L_start, L_end + 1):
                # Check if all divisors have at least one multiple in [L, L+W]
                available = []
                valid = True
                for i, d in enumerate(divisors):
                    k = (L + d - 1) // d
                    first = k * d
                    if first > L + W:
                        valid = False
                        break
                    multiples = []
                    x = first
                    while x <= L + W:
                        multiples.append(x)
                        x += d
                    available.append(multiples)
                
                if not valid:
                    continue
                
                # Backtracking
                indices = sorted(range(n), key=lambda i: len(available[i]))
                used = set()
                choices = [0] * n
                
                def backtrack(idx):
                    if idx == n:
                        return True
                    i = indices[idx]
                    for val in available[i]:
                        if val not in used:
                            used.add(val)
                            choices[i] = val
                            if backtrack(idx + 1):
                                return True
                            used.remove(val)
                    return False
                
                if backtrack(0):
                    result = choices[:]
                    # Verify
                    if len(set(result)) == n and all(0 < x <= 1000000 for x in result) and all(result[i] % divisors[i] == 0 for i in range(n)):
                        best_W = W
                        best_result = [int(x) for x in result]
                        found = True
                        break
            if found:
                break
        
        if found:
            break

    if best_result:
        return best_result
    
    # Fallback to original logic if no small W found
    # ... (keep original logic as fallback)
    best_W = 33
    for L in range(1, LCM + 1):
        if time.time() - start_time > timeout:
            break
        firsts = []
        for d in divisors:
            k = (L + d - 1) // d
            firsts.append(k * d)
        min_W = max(firsts) - L
        if min_W >= best_W:
            continue
        for W in range(min_W, best_W):
            available = []
            valid = True
            for d in divisors:
                k = (L + d - 1) // d
                first = k * d
                if first > L + W:
                    valid = False
                    break
                multiples = []
                x = first
                while x <= L + W:
                    multiples.append(x)
                    x += d
                available.append(multiples)
            if not valid:
                continue
            indices = sorted(range(n), key=lambda i: len(available[i]))
            used = set()
            choices = [0] * n
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for val in available[i]:
                    if val not in used:
                        used.add(val)
                        choices[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            if backtrack(0):
                result = choices[:]
                if len(set(result)) == n and all(0 < x <= 1000000 for x in result) and all(result[i] % divisors[i] == 0 for i in range(n)):
                    best_W = W
                    best_result = [int(x) for x in result]
                    break
        if best_W == 9:
            break

    if best_result:
        return best_result
    return [int(x) for x in divisors]
