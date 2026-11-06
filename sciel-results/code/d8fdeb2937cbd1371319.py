import math

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # Known valid solution with cardinality 9 (maximum possible for 3x3 sumsets)
    # A = [18, 42, 50], B = [1, 11, 17]
    # Sums: 19, 29, 35, 43, 53, 59, 51, 61, 67
    # All 9 distinct, all pairwise coprime
    A = [18, 42, 50]
    B = [1, 11, 17]
    
    # Verify validity
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    
    # Check pairwise coprime
    valid = True
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if math.gcd(distinct[i], distinct[j]) > 1:
                valid = False
                break
        if not valid:
            break
    
    if valid and len(distinct) == 9:
        return [A, B]
    
    # Fallback: search for a valid solution
    # This should never be reached with the known solution above
    import random
    random.seed(42)
    all_vals = list(range(1, 129))
    
    best_result = [[1, 7, 13], [4, 10, 16]]
    best_count = 0
    
    def score_pair(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        count = len(distinct)
        coprime = True
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    coprime = False
                    break
            if not coprime:
                break
        return count, coprime
    
    for _ in range(100000):
        A_try = random.sample(all_vals, 3)
        B_try = random.sample(all_vals, 3)
        count, coprime = score_pair(A_try, B_try)
        if coprime and count > best_count:
            best_count = count
            best_result = [sorted(A_try), sorted(B_try)]
            if best_count == 9:
                break
    
    return best_result
