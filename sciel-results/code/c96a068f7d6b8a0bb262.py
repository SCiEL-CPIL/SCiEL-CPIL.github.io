def construct():
    N = 256
    candidates = list(range(2, N + 1))
    S = []
    
    while candidates:
        best_candidate = None
        min_conflicts = float('inf')
        
        for x in candidates:
            conflicts = 0
            # Check against all pairs in S
            for i in range(len(S)):
                b = S[i]
                for j in range(i + 1, len(S)):
                    c = S[j]
                    if (b * c) % x == 0 or (x * c) % b == 0 or (x * b) % c == 0:
                        conflicts += 1
                        # Early break? If we just need to know if it's 0, we can break on first conflict.
                        # But we want the MINIMUM conflicts. So we must count all?
                        # Actually, if conflicts > 0, we can't add x.
                        # We only care about candidates with 0 conflicts.
                        # So if we find 1 conflict, we can stop checking this x?
                        # Yes, because if conflicts > 0, x is not a valid addition.
                        # We are looking for ANY x with 0 conflicts.
                        break
                if conflicts > 0:
                    break
            
            if conflicts == 0:
                # Found a valid candidate.
                # Heuristic: pick the one that is "best" among valid ones.
                # e.g., largest value, or smallest prime factor?
                # Let's pick the largest value to leave smaller numbers for later?
                # Or smallest value?
                # Usually, larger numbers are "safer" as divisors (harder to divide products).
                # But they are "safer" as multiplicands too?
                # Let's try picking the largest valid candidate.
                if best_candidate is None or x > best_candidate:
                    best_candidate = x
                    min_conflicts = 0
        
        if best_candidate is not None:
            S.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            # No candidate has 0 conflicts.
            # We are stuck. 
            # We could try a different heuristic or stop.
            # Stopping yields a valid set.
            break
            
    return [int(x) for x in S]
