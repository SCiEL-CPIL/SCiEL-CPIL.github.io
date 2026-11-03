import itertools
import math

def construct():
    best_score = 0
    best_set = [0, 1, 5, 8]
    
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        
        sum_mask = 0
        diff_mask = 0
        
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    # Search parameters
    # M: max element
    # n: size of set
    # We search M from 4 to 30
    for M in range(4, 31):
        # Determine max n based on M to keep combination count reasonable
        if M <= 10:
            max_n = 10
        elif M <= 15:
            max_n = 9
        elif M <= 20:
            max_n = 8
        elif M <= 25:
            max_n = 7
        else:
            max_n = 6
            
        for n in range(3, max_n + 1):
            if n - 1 > M:
                continue
                
            for comb in itertools.combinations(range(1, M + 1), n - 1):
                u = [0] + list(comb)
                s = calc_score(u)
                if s > best_score:
                    best_score = s
                    best_set = u

    return best_set
