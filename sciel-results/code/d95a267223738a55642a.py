import random
import itertools
import time

def construct():
    def is_valid_fast(S):
        n = len(S)
        sums = set()
        for r in range(1, n + 1):
            for combo in itertools.combinations(S, r):
                s = sum(combo)
                if s in sums:
                    return False
                sums.add(s)
        
        sum_list = sorted(sums)
        for i in range(len(sum_list)):
            a = sum_list[i]
            for j in range(i + 1, len(sum_list)):
                if sum_list[j] % a == 0:
                    return False
        return True

    start_time = time.time()
    time_limit = 50

    for target_size in range(8, 3, -1):
        if time.time() - start_time > time_limit:
            break
        
        attempts = 5000 if target_size <= 7 else 2000
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
            if random.random() < 0.5:
                S = random.sample(range(100, 257), target_size)
            else:
                S = random.sample(range(1, 257), target_size)
            if is_valid_fast(S):
                return list(S)
    
    return [31, 32, 34]
