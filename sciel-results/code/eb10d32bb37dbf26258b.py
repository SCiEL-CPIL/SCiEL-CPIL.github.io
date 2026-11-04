import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin
    
    # Candidate sets
    # Based on analysis, {0, 1, 3, 7, 15} performs well
    candidates = [
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7],
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9, 25],
        [0, 1, 3, 9, 26],
        [0, 1, 3, 9, 28],
        [0, 1, 3, 9, 29],
        [0, 1, 3, 9, 30],
        [0, 1, 3, 9, 31],
    ]
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    def score_set(U):
        if not U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        # Calculate sumset and differenceset sizes
        # Using sets for efficiency
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1:
            return 1.0 # log(1)=0
        
        # log2(ratio) / log2(2*max_u + 1)
        numerator = math.log2(ratio)
        denominator = math.log2(2 * max_u + 1)
        
        return 1.0 + numerator / denominator

    for U in candidates:
        if time.time() > deadline:
            break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set
