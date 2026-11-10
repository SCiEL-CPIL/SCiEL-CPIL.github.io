import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of the divisors
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    LCM = 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # We know W=33 is possible (the divisors themselves). Let's try to find something smaller.
    # The minimum possible W is at least 9 (since we need 10 distinct integers).
    # But due to divisibility constraints, it's likely larger.
    
    # For a fixed W, we need to find L such that there exist distinct multiples m_i of d_i all in [L, L+W].
    # Since the pattern repeats every LCM, we only need to check L in [1, LCM].
    
    # Optimization: Precompute for each divisor d and each residue r (0 <= r < d),
    # whether there is a multiple of d in [L, L+W] for L ≡ r (mod d).
    # Actually, for a fixed W, the condition depends on L mod d.
    # Let's precompute for each W and each d, the set of residues r (mod d) such that
    # there is a multiple of d in [L, L+W] for L ≡ r (mod d).
    
    # But it's simpler to just check on the fly.
    
    # Let's use a smarter search:
    # For each W, iterate L from 1 to LCM.
    # For each L, check if a valid assignment exists using backtracking.
    
    # To speed up the backtracking:
    # 1. For each divisor d_i, compute the list of multiples in [L, L+W].
    # 2. Sort divisors by the number of available multiples (ascending) to find conflicts early.
    # 3. Use a bitmask for "used" values if W < 63.
    
    def check_window(W, L):
        # Returns a list of chosen values or None if no solution
        # We use a bitmask for used values if W < 63
        
        # Precompute candidates for each divisor
        candidates = []
        for d in divisors:
            k = (L + d - 1) // d
            first = k * d
            if first > L + W:
                return None
            # Generate candidates as offsets from L
            cand = []
            x = first
            while x <= L + W:
                cand.append(x - L)  # Store offset
                x += d
            candidates.append(cand)
        
        # If any candidate list is empty, return None (already checked first > L+W)
        # Sort by length
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        # Backtracking
        # If W < 63, use bitmask
        use_bitmask = W < 63
        
        if use_bitmask:
            used = 0
            choices = [0] * n
            
            def bt(idx):
                nonlocal used
                if idx == n:
                    return True
                
                i = indices[idx]
                for off in candidates[i]:
                    bit = 1 << off
                    if not (used & bit):
                        used |= bit
                        choices[i] = off
                        if bt(idx + 1):
                            return True
                        used &= ~bit
                
                return False
            
            if bt(0):
                return [L + choices[i] for i in range(n)]
            else:
                return None
        else:
            used = set()
            choices = [0] * n
            
            def bt(idx):
                if idx == n:
                    return True
                
                i = indices[idx]
                for off in candidates[i]:
                    val = L + off
                    if val not in used:
                        used.add(val)
                        choices[i] = off
                        if bt(idx + 1):
                            return True
                        used.remove(val)
                
                return False
            
            if bt(0):
                return [L + choices[i] for i in range(n)]
            else:
                return None
    
    # Iterate W from 9 to 33 (current best)
    for W in range(9, 34):
        if time.time() - start_time > timeout:
            break
        
        # For each L in [1, LCM], check if a valid assignment exists.
        # Optimization: Precompute for each divisor d, the set of residues r (mod d)
        # such that there is a multiple of d in [L, L+W].
        # Actually, let's just check on the fly.
        
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                break
            
            # Quick check: for each d, is there a multiple in [L, L+W]?
            feasible = True
            for d in divisors:
                # Check if interval [L, L+W] contains a multiple of d
                # This is true if ceil(L/d) * d <= L + W
                if (L + d - 1) // d * d > L + W:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            result = check_window(W, L)
            if result is not None:
                # Verify
                if len(set(result)) == n:
                    if all(result[i] % divisors[i] == 0 for i in range(n)):
                        if all(1 <= x <= 1000000 for x in result):
                            return result
    
    # If no improvement found, return the divisors themselves
    return divisors[:]
