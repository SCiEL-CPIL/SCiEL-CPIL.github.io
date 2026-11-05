import time

P = 1009
M = 5
DEADLINE_MARGIN = 2.0  # seconds to stop before the 90s wall-clock limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    # Build a boolean array is_qr where is_qr[v] is True if v is a nonzero quadratic residue mod P.
    # Also store one root for each residue for the output.
    is_qr = [False] * P
    roots = [0] * P  # roots[v] = x such that x^2 % P == v, 0 if not a QR
    
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            # We only need one root. The first one found is fine.
            # Storing it allows us to return it without searching again.
            if roots[r] == 0:
                roots[r] = x

    # Search for the minimum s such that s, s+1, ..., s+4 are all nonzero QRs.
    # s must be in [1, P - M] to ensure s+4 <= P-1 (no wraparound, no zero).
    # Range of s: 1 to 1009 - 5 = 1004.
    
    # Optimization: Precompute a "run" array?
    # For P=1009, simple loop is sufficient and fast enough.
    # is_qr is a list of booleans. Access is O(1).
    
    for s in range(1, P - M + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        # Unroll the loop for speed, though not strictly necessary for P=1009
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            # Found the minimal s
            result_roots = [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            
            # Validation
            for i in range(M):
                r = result_roots[i]
                if (r * r) % P != s + i:
                    # Fallback: find a valid root if the stored one is somehow invalid (shouldn't happen)
                    # Search for a root
                    found = False
                    for x in range(1, P):
                        if (x * x) % P == s + i:
                            result_roots[i] = x
                            found = True
                            break
                    if not found:
                        raise RuntimeError(f"No root found for {s+i}")
            
            # Check deadline before returning? 
            # The task says "Keep an EXPLICIT deadline guard... return the best result found".
            # Since we found the *minimal* s, it is the best result.
            # We can return immediately.
            return {'start': s, 'roots': result_roots}
            
        # Check deadline
        if time.time() > deadline:
            break

    # If we didn't find a run in the first pass (or timed out), 
    # we should ideally return the best found so far? 
    # But the task asks for a valid run. If none found, we might have to search harder?
    # Given the density, a run must exist.
    # If we timed out, we might have missed the first one? No, we iterate from 1.
    # If we broke due to deadline, we didn't find any.
    # This is highly unlikely for P=1009.
    
    # Fallback: Search again without deadline guard if we failed?
    # Or just return a known valid late run if we have one?
    # We don't store candidates.
    # Let's just raise an error or return a dummy if this happens, 
    # but for P=1009, it will find s=1 instantly.
    
    raise RuntimeError("No run found within deadline or search space")
