"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(S_set, S_list, x):
    """Check if x can be added to S without violating constraint."""
    # Need: no distinct a,b in S such that a+b in S, a+x in S, b+x in S
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        ax = a + x
        if ax not in S_set:
            continue
        for j in range(i+1, n):
            b = S_list[j]
            bx = b + x
            if bx not in S_set:
                continue
            if (a + b) in S_set:
                return False
    return True

def _greedy(n, order):
    S = []
    S_set = set()
    for x in order:
        if _can_add(S_set, S, x):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _verify(S):
    """Verify the set is valid."""
    S_set = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S[k]
                if (a + c) in S_set and (b + c) in S_set:
                    return False
    return True

def construct():
    n = 128
    t0 = time.time()
    deadline = t0 + 80.0
    
    random.seed(42)
    
    # Try multiple greedy strategies
    best_S = []
    best_size = 0
    
    # Strategy 1: upper half
    S1, _ = _greedy(n, list(range(65, 129)))
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Strategy 2: all odds
    S2, _ = _greedy(n, list(range(1, 129, 2)))
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Strategy 3: random orders
    for trial in range(100):
        if time.time() > deadline - 40:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        S3, _ = _greedy(n, order)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)
    
    # Local search
    if time.time() < deadline - 15:
        cur = list(best_S)
        cur_set = set(cur)
        cur_size = len(cur)
        candidates = [x for x in range(1, n+1) if x not in cur_set]
        
        for it in range(500000):
            if time.time() > deadline - 5:
                break
            
            if candidates:
                idx = random.randint(0, len(candidates)-1)
                x = candidates[idx]
                if _can_add(cur_set, cur, x):
                    cur.append(x)
                    cur_set.add(x)
                    cur_size += 1
                    candidates.pop(idx)
                    if cur_size > best_size:
                        best_size = cur_size
                        best_S = list(cur)
                    continue
            
            # Swap: remove random, try to add
            if cur_size > 0:
                ri = random.randint(0, cur_size-1)
                rem = cur[ri]
                cur.pop(ri)
                cur_set.discard(rem)
                cur_size -= 1
                
                # Try to add from candidates
                added = False
                if candidates:
                    random.shuffle(candidates)
                    for x in candidates[:30]:
                        if _can_add(cur_set, cur, x):
                            cur.append(x)
                            cur_set.add(x)
                            cur_size += 1
                            candidates.remove(x)
                            added = True
                            break
                
                if cur_size > best_size:
                    best_size = cur_size
                    best_S = list(cur)
                elif not added:
                    # Revert
                    cur.append(rem)
                    cur_set.add(rem)
                    cur_size += 1
                    if rem not in cur_set:  # should be in
                        pass
                    # rem is back in cur, so it's not a candidate
                    # But we need to make sure candidates doesn't have rem
                    # It shouldn't since we discarded it
                    pass
    
    # Final verification
    if not _verify(best_S):
        # Fallback
        best_S = list(range(65, 129))
    
    return list(best_S)
# EVOLVE-BLOCK-END
