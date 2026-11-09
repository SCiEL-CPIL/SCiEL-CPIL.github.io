import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    MAX_VAL = 128
    MAX_SUM = 256  # max possible sum
    
    # Precompute coprime table: coprime[i][j] = True if gcd(i,j)==1
    # For efficiency, we'll use a set of "bad" pairs or a 2D list
    # Since max sum is 256, we need coprime checks up to 256
    coprime = [[True] * (MAX_SUM + 1) for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1):
            if math.gcd(i, j) != 1:
                coprime[i][j] = False
                coprime[j][i] = False
    
    # For a set of numbers, check if all pairwise coprime using precomputed table
    def all_pairwise_coprime(nums):
        """nums is a list/tuple of ints, all in [1, MAX_SUM]"""
        n = len(nums)
        for i in range(n):
            ni = nums[i]
            for j in range(i + 1, n):
                nj = nums[j]
                if not coprime[ni][nj]:
                    return False
        return True
    
    # Known good: score 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    def search_for_B(A, max_val, current_best):
        a1, a2, a3 = A[0], A[1], A[2]
        best_B = None
        best_s = current_best
        
        # Precompute valid b values for each column
        # For column j with value aj, the sums are aj + b
        # We need all 9 sums to be pairwise coprime
        
        # Iterate b1, b2, b3 with b1 < b2 < b3
        for b1 in range(1, max_val + 1):
            s11 = a1 + b1
            s21 = a2 + b1
            s31 = a3 + b1
            
            # Check column 1 internal
            if not coprime[s11][s21] or not coprime[s11][s31] or not coprime[s21][s31]:
                continue
            
            col1 = (s11, s21, s31)
            
            for b2 in range(b1 + 1, max_val + 1):
                s12 = a1 + b2
                s22 = a2 + b2
                s32 = a3 + b2
                
                # Check column 2 internal
                if not coprime[s12][s22] or not coprime[s12][s32] or not coprime[s22][s32]:
                    continue
                
                # Check column 2 vs column 1
                ok = True
                for ns in (s12, s22, s32):
                    for ps in col1:
                        if not coprime[ns][ps]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                col12 = col1 + (s12, s22, s32)
                
                for b3 in range(b2 + 1, max_val + 1):
                    s13 = a1 + b3
                    s23 = a2 + b3
                    s33 = a3 + b3
                    
                    # Check column 3 internal
                    if not coprime[s13][s23] or not coprime[s13][s33] or not coprime[s23][s33]:
                        continue
                    
                    # Check column 3 vs previous
                    ok = True
                    for ns in (s13, s23, s33):
                        for ps in col12:
                            if not coprime[ns][ps]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    all_sums = col12 + (s13, s23, s33)
                    score = len(set(all_sums))
                    if score > best_s:
                        best_s = score
                        best_B = [b1, b2, b3]
                        if best_s >= 9:
                            return best_B, best_s
        return best_B, best_s
    
    # Phase 1: small A values - exhaustive
    for a1 in range(1, 30):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 30):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 30):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 30, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    if time.time() > deadline:
        return [list(best_A), list(best_B)]
    
    # Phase 2: medium
    for a1 in range(1, 50):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 50, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    if time.time() > deadline:
        return [list(best_A), list(best_B)]
    
    # Phase 3: larger
    for a1 in range(1, 80):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 80):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 80):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                best_B, score = search_for_B(A, 80, best_score)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = best_B
                    if best_score >= 9:
                        return [list(best_A), list(best_B)]
    
    return [list(best_A), list(best_B)]
