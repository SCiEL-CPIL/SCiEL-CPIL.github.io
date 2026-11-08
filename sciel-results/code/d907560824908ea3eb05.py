import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_score = 0.0
    best_set = [0]
    
    # Exhaustive search for small M
    # U must contain 0. So we iterate over subsets of {1, ..., M}
    # and add 0.
    
    for M in range(1, 21):
        if time.time() > deadline:
            break
            
        # Iterate over all subsets of {1, ..., M}
        # There are 2^M such subsets.
        # We can iterate from 0 to 2^M - 1
        # Bit i set means (i+1) is in the subset.
        
        for mask in range(1 << M):
            if time.time() > deadline:
                return best_set
                
            U = [0]
            for i in range(M):
                if mask & (1 << i):
                    U.append(i + 1)
                    
            # Calculate Score
            max_u = max(U)
            if max_u == 0:
                # U = [0]
                # S={0}, D={0}. Ratio 1. Score 1.
                score = 1.0
            else:
                sums = set()
                diffs = set()
                for a in U:
                    for b in U:
                        sums.add(a + b)
                        diffs.add(a - b)
                        
                len_s = len(sums)
                len_d = len(diffs)
                
                if len_s == 0:
                    score = 0.0
                else:
                    ratio = len_d / len_s
                    if ratio <= 1:
                        score = 1.0 # log(1)=0
                    else:
                        score = 1.0 + math.log2(ratio) / math.log2(2 * max_u + 1)
            
            if score > best_score:
                best_score = score
                best_set = U.copy()
                
    # If time remains, do some random search for larger M
    if time.time() < deadline - 5:
        for _ in range(1000):
            if time.time() > deadline:
                break
            M = random.randint(20, 50)
            size = random.randint(3, 10)
            # Pick random elements from 1..M
            candidates = random.sample(range(1, M + 1), size)
            U = [0] + sorted(candidates)
            
            max_u = max(U)
            sums = set()
            diffs = set()
            for a in U:
                for b in U:
                    sums.add(a + b)
                    diffs.add(a - b)
                    
            len_s = len(sums)
            len_d = len(diffs)
            ratio = len_d / len_s
            score = 1.0 + math.log2(ratio) / math.log2(2 * max_u + 1)
            
            if score > best_score:
                best_score = score
                best_set = U.copy()
                
    return best_set
