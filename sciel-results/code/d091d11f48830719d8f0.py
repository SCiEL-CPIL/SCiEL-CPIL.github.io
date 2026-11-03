import math
import itertools

def construct():
    def is_golomb(U):
        U = sorted(U)
        diffs = set()
        for i in range(len(U)):
            for j in range(i+1, len(U)):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        # Calculate |U-U| and |U+U|
        # For efficiency, use sets
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
        
        n_sums = len(sums)
        n_diffs = len(diffs)
        
        if n_sums == 0:
            return 1.0
            
        ratio = n_diffs / n_sums
        if ratio <= 1:
            return 1.0
        
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    best_U = [0]
    best_score = -1
    
    # Optimal Golomb ruler lengths for N=1..10
    # 1: 0
    # 2: 1
    # 3: 3
    # 4: 6
    # 5: 11
    # 6: 17
    # 7: 28
    # 8: 37
    # 9: 49
    # 10: 63
    max_lengths = {
        1: 0,
        2: 1,
        3: 3,
        4: 6,
        5: 11,
        6: 17,
        7: 28,
        8: 37,
        9: 49,
        10: 63
    }
    
    for N in range(1, 9):
        L_max = max_lengths[N] + 5 # Allow some slack for non-optimal length but better ratio?
        # Actually, if L is larger, denominator is larger. 
        # But if L is smaller, it might not be a Golomb ruler.
        # We only care about Golomb rulers.
        # If we restrict to L <= L_optimal, we only check the optimal ones.
        # But maybe a slightly longer ruler has better sum properties?
        # Let's check up to L_optimal + 10.
        
        if N == 1:
            U = [0]
            s = score(U)
            if s > best_score:
                best_score = s
                best_U = U
            continue

        # Generate combinations
        # To speed up, we can prune.
        # But for N<=8, it's fast enough.
        
        # We need to pick N-1 numbers from 1..L_max
        # such that they form a Golomb ruler with 0.
        
        # Optimization: Use backtracking to generate Golomb rulers directly?
        # Or just filter combinations.
        
        # For N=8, L=37. Combinations C(37, 7) is ~ 10 million.
        # This might be too slow in Python (10-20s).
        # Let's reduce range to L_optimal + 2.
        # Or use N up to 7.
        
        # Let's try N up to 7 with L_optimal + 5.
        # N=7, L=28. C(33, 6) ~ 1.1 million. Fast.
        # N=8, L=37. C(42, 7) ~ 26 million. Too slow.
        
        # So we will brute force N=1..7.
        # For N=8..10, we will use the known optimal rulers from the previous code,
        # but also check a few variations?
        # Or just rely on the fact that score drops.
        
        if N > 7:
            continue
            
        # Use itertools.combinations
        # We can prune: if the last element is too small, we can't reach L?
        # No, we are just iterating.
        
        # To make it faster, we can check Golomb property incrementally?
        # No, simple filter is fine for N<=7.
        
        for comb in itertools.combinations(range(1, L_max + 1), N - 1):
            U = [0] + list(comb)
            if is_golomb(U):
                s = score(U)
                if s > best_score:
                    best_score = s
                    best_U = U
                    
    # Also check the hardcoded rulers for N=8,9,10 from the previous program
    # in case they are better (unlikely, but safe)
    rulers = {
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 41, 47]
    }
    
    for N in range(8, 11):
        U = rulers[N]
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
            
    return best_U
