import random
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s
    
    def elapsed():
        return time.time() - start_time

    def can_add(x, S, sums_mask, diff_mask_self, diff_mask_other):
        # Check Sidon: 2x not in sums, x+s not in sums
        if (sums_mask >> (2*x)) & 1:
            return False, None
        for s in S:
            if (sums_mask >> (x+s)) & 1:
                return False, None
        
        # Check Differences
        new_diffs = []
        for s in S:
            d = abs(x - s)
            if d == 0:
                continue
            # d must not be in other set's diffs
            if (diff_mask_other >> d) & 1:
                return False, None
            # d must not be in self set's diffs (Sidon property implies unique diffs)
            if (diff_mask_self >> d) & 1:
                return False, None
            new_diffs.append(d)
        
        return True, new_diffs

    def greedy_build(candidates, diff_mask_other, target_size):
        S = []
        sums_mask = 0
        diff_mask_self = 0
        
        local_candidates = candidates[:]
        random.shuffle(local_candidates)
        
        for x in local_candidates:
            if len(S) >= target_size:
                break
            ok, new_diffs = can_add(x, S, sums_mask, diff_mask_self, diff_mask_other)
            if ok:
                S.append(x)
                # Update sums_mask
                sums_mask |= (1 << (2*x))
                for s in S[:-1]:
                    sums_mask |= (1 << (x+s))
                # Update diff_mask_self
                for d in new_diffs:
                    diff_mask_self |= (1 << d)
                    
        return S, sums_mask, diff_mask_self

    def local_search(A, B, sums_A, diff_A, sums_B, diff_B):
        # Try to add to A
        for x in range(1, n + 1):
            if x in A:
                continue
            ok, new_diffs = can_add(x, A, sums_A, diff_A, diff_B)
            if ok:
                A.append(x)
                sums_A |= (1 << (2*x))
                for s in A[:-1]:
                    sums_A |= (1 << (x+s))
                for d in new_diffs:
                    diff_A |= (1 << d)
        
        # Try to add to B
        for x in range(1, n + 1):
            if x in B:
                continue
            ok, new_diffs = can_add(x, B, sums_B, diff_B, diff_A)
            if ok:
                B.append(x)
                sums_B |= (1 << (2*x))
                for s in B[:-1]:
                    sums_B |= (1 << (x+s))
                for d in new_diffs:
                    diff_B |= (1 << d)
                    
        return A, B, sums_A, diff_A, sums_B, diff_B

    best_score = 0
    best_A = [1]
    best_B = [2]
    
    candidates = list(range(1, n + 1))
    
    # Multiple restarts
    for _ in range(10000):
        if elapsed() > deadline:
            break
        
        # Build A
        A, sums_A, diff_A = greedy_build(candidates, 0, 20)
        if not A:
            continue
        
        # Build B, constrained by A's diffs
        B, sums_B, diff_B = greedy_build(candidates, diff_A, 20)
        if not B:
            continue
        
        # Local search
        A, B, sums_A, diff_A, sums_B, diff_B = local_search(A, B, sums_A, diff_A, sums_B, diff_B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            
            if best_score >= 32:
                break

    return [best_A, best_B]
