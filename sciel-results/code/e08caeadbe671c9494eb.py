import math
import time

def construct():
    # Known optimal Golomb Rulers for small n
    # n=3: [0, 1, 3]
    # n=4: [0, 1, 4, 10]
    # n=5: [0, 1, 4, 9, 11]
    # n=6: [0, 1, 4, 10, 12, 17]
    # n=7: [0, 1, 4, 10, 18, 23, 25]
    # n=8: [0, 1, 4, 10, 18, 23, 29, 34]
    # n=9: [0, 1, 4, 10, 18, 23, 29, 36, 40]
    # n=10: [0, 1, 4, 10, 18, 23, 29, 36, 45, 52]
    
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 10],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 36, 40],
        [0, 1, 4, 10, 18, 23, 29, 36, 45, 52]
    ]
    
    best_set = candidates[0]
    best_score = -1
    
    for U in candidates:
        if not U or U[0] != 0:
            continue
            
        max_u = max(U)
        if max_u == 0:
            score = 1.0
        else:
            # Compute |U-U| and |U+U|
            diffs = set()
            sums = set()
            n = len(U)
            for i in range(n):
                for j in range(n):
                    diffs.add(U[i] - U[j])
                    sums.add(U[i] + U[j])
            
            size_diff = len(diffs)
            size_sum = len(sums)
            
            if size_sum == 0:
                score = 1.0
            else:
                ratio = size_diff / size_sum
                # log base 2 or e doesn't matter for ratio, but formula uses log
                # The prompt says "log", usually natural log or base 2.
                # The ratio of logs is invariant to base.
                denom = math.log(2 * max_u + 1)
                if denom == 0:
                    score = 1.0
                else:
                    score = 1 + math.log(ratio) / denom
        
        if score > best_score:
            best_score = score
            best_set = U
            
    return best_set
