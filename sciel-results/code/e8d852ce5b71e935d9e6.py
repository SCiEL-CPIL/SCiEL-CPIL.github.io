import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    MAX_VAL = 128
    MAX_SUM = 256
    
    # Precompute coprime table as bitsets for fast intersection
    # coprime_bits[i] is a bitmask where bit j is set if gcd(i,j)==1
    coprime_bits = [0] * (MAX_SUM + 1)
    for i in range(1, MAX_SUM + 1):
        mask = 0
        for j in range(1, MAX_SUM + 1):
            if math.gcd(i, j) == 1:
                mask |= (1 << j)
        coprime_bits[i] = mask
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    def is_pairwise_coprime(sums):
        """Check if all distinct values in sums are pairwise coprime."""
        distinct = list(set(sums))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Return (score, valid) for a pair (A, B)."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return 0, False
        return len(distinct), True
    
    def search_for_B(A):
        """Find best B for given A using bitset acceleration."""
        a1, a2, a3 = A[0], A[1], A[2]
        best_B = None
        best_s = 0
        
        # For each b, compute the set of sums {a1+b, a2+b, a3+b}
        # and check if they are pairwise coprime
        valid_b = []
        for b in range(1, MAX_VAL + 1):
            s1, s2, s3 = a1 + b, a2 + b, a3 + b
            # Check pairwise coprime for this column
            if (coprime_bits[s1] & (1 << s2)) and \
               (coprime_bits[s1] & (1 << s3)) and \
               (coprime_bits[s2] & (1 << s3)):
                valid_b.append(b)
        
        n_valid = len(valid_b)
        if n_valid < 3:
            return None, 0
        
        # Try all triples from valid_b
        # Use bitset to check cross-compatibility
        for i in range(n_valid):
            b1 = valid_b[i]
            s1a, s1b, s1c = a1 + b1, a2 + b1, a3 + b1
            for j in range(i + 1, n_valid):
                b2 = valid_b[j]
                s2a, s2b, s2c = a1 + b2, a2 + b2, a3 + b2
                # Check cross-compatibility between col1 and col2
                ok = True
                for ns in (s2a, s2b, s2c):
                    for ps in (s1a, s1b, s1c):
                        if not (coprime_bits[ns] & (1 << ps)):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for k in range(j + 1, n_valid):
                    b3 = valid_b[k]
                    s3a, s3b, s3c = a1 + b3, a2 + b3, a3 + b3
                    # Check cross-compatibility between col3 and cols 1,2
                    ok = True
                    for ns in (s3a, s3b, s3c):
                        for ps in (s1a, s1b, s1c, s2a, s2b, s2c):
                            if not (coprime_bits[ns] & (1 << ps)):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    all_sums = [s1a, s1b, s1c, s2a, s2b, s2c, s3a, s3b, s3c]
                    score = len(set(all_sums))
                    if score > best_s:
                        best_s = score
                        best_B = [b1, b2, b3]
                        if best_s >= 9:
                            return best_B, best_s
        return best_B, best_s
    
    # Generate candidate A values
    # Key insight: A values that are themselves "spread out" and avoid
    # creating sums with common factors are more promising
    candidate_As = []
    for a1 in range(1, 60):
        for a2 in range(a1 + 1, min(a1 + 40, 60)):
            for a3 in range(a2 + 1, min(a2 + 30, 60)):
                A = [a1, a2, a3]
                # Quick filter: count how many b values give valid columns
                cvb = 0
                for b in range(1, 129):
                    s1, s2, s3 = a1 + b, a2 + b, a3 + b
                    if (coprime_bits[s1] & (1 << s2)) and \
                       (coprime_bits[s1] & (1 << s3)) and \
                       (coprime_bits[s2] & (1 << s3)):
                        cvb += 1
                if cvb >= 5:
                    candidate_As.append((cvb, A))
    
    candidate_As.sort(key=lambda x: -x[0])
    
    for cvb, A in candidate_As[:500]:
        if time.time() > deadline:
            break
        best_B, score = search_for_B(A)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = best_B
            if best_score >= 9:
                return [list(best_A), list(best_B)]
    
    # Broader search if needed
    if best_score < 9 and time.time() < deadline - 10:
        for a1 in range(1, 80):
            if time.time() > deadline - 5:
                break
            for a2 in range(a1 + 1, min(a1 + 50, 80)):
                if time.time() > deadline - 5:
                    break
                for a3 in range(a2 + 1, min(a2 + 40, 80)):
                    if time.time() > deadline - 5:
                        break
                    A = [a1, a2, a3]
                    best_B, score = search_for_B(A)
                    if score > best_score:
                        best_score = score
                        best_A = A
                        best_B = best_B
                        if best_score >= 9:
                            return [list(best_A), list(best_B)]
    
    # Final verification
    if best_B is not None:
        s, valid = score_pair(best_A, best_B)
        if valid and s > best_score:
            best_score = s
    
    return [list(best_A), list(best_B)]
