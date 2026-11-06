"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid_set(S):
    """Check if set S satisfies the constraint."""
    S_list = sorted(S)
    for i, a in enumerate(S_list):
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Greedy top-down construction
    S = set()
    
    for cand in range(n, 0, -1):
        if time.time() > deadline:
            break
        valid = True
        
        # Case 1: cand plays role of 'a' (smallest in triple)
        # Need: no b,c in S with b > cand, c >= b, cand | (b+c)
        larger = [x for x in S if x > cand]
        for i in range(len(larger)):
            b = larger[i]
            for j in range(i, len(larger)):
                c = larger[j]
                if (b + c) % cand == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Case 2: cand plays role of 'b' (middle in triple)
            # Need: no a,c in S with a < cand, c >= cand, a | (cand + c)
            smaller = [x for x in S if x < cand]
            geq = [x for x in S if x >= cand]
            for a in smaller:
                for c in geq:
                    if (cand + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Case 3: cand plays role of 'c' (largest in triple)
            # Need: no a,b in S with a < b <= cand, a | (b + cand)
            for a in S:
                if a >= cand:
                    continue
                for b in S:
                    if a < b <= cand and (b + cand) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            S.add(cand)
    
    # Local search: try to add elements that were skipped
    S_list = sorted(S)
    skipped = set(range(1, n+1)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in sorted(skipped, reverse=True):
            if time.time() > deadline:
                break
            valid = True
            
            # Case 1: cand is 'a'
            larger = [x for x in S if x > cand]
            for i in range(len(larger)):
                b = larger[i]
                for j in range(i, len(larger)):
                    c = larger[j]
                    if (b + c) % cand == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Case 2: cand is 'b'
                smaller = [x for x in S if x < cand]
                geq = [x for x in S if x >= cand]
                for a in smaller:
                    for c in geq:
                        if (cand + c) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Case 3: cand is 'c'
                for a in S:
                    if a >= cand:
                        continue
                    for b in S:
                        if a < b <= cand and (b + cand) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                S.add(cand)
                skipped.discard(cand)
                improved = True
    
    return sorted(S)
# EVOLVE-BLOCK-END
