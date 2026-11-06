import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = []
    
    def try_add(x, S, used_sums):
        new_sums = set()
        s = 3 * x
        if s in used_sums:
            return False
        new_sums.add(s)
        
        for y in S:
            s1 = 2 * x + y
            if s1 in used_sums or s1 in new_sums:
                return False
            new_sums.add(s1)
            
            s2 = x + 2 * y
            if s2 in used_sums or s2 in new_sums:
                return False
            new_sums.add(s2)
            
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                s3 = x + y + z
                if s3 in used_sums or s3 in new_sums:
                    return False
                new_sums.add(s3)
                
        used_sums.update(new_sums)
        S.append(x)
        return True

    # Strategy 1: Pure Random
    # Strategy 2: Geometric Start
    
    for seed in range(5000):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        S = []
        used_sums = set()
        
        mode = seed % 2
        if mode == 0:
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        else:
            # Geometric start
            geo = [1, 3, 9, 27, 81, 243, 729, 2187]
            # Check if geo is valid (it is)
            for g in geo:
                try_add(g, S, used_sums)
            # Fill with rest
            rest = [x for x in range(1, 4097) if x not in geo]
            random.shuffle(rest)
            candidates = rest
            
        for x in candidates:
            if len(S) > len(best_set):
                # Update best set incrementally? No, just at end of seed
                pass
            if len(S) >= 20:
                break
            try_add(x, S, used_sums)
        
        if len(S) > len(best_set):
            best_set = list(S)
            if len(best_set) >= 20:
                break

    return best_set
