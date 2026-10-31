import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Search for small sets
    # n = 3, 4, 5
    # max_val = 20
    
    for n in range(3, 7):
        if time.time() > deadline:
            break
        for max_val in range(n, 30):
            if time.time() > deadline:
                break
            # Generate subsets of {1, ..., max_val} of size n-1
            # Always include 0
            candidates = list(range(1, max_val + 1))
            for combo in itertools.combinations(candidates, n - 1):
                U = [0] + list(combo)
                U.sort()
                
                # Calculate score
                sumset = set()
                diffset = set()
                for a in U:
                    for b in U:
                        sumset.add(a + b)
                        diffset.add(a - b)
                
                size_sum = len(sumset)
                size_diff = len(diffset)
                max_u = max(U)
                
                if size_sum == 0:
                    continue
                    
                ratio = size_diff / size_sum
                if ratio <= 1:
                    continue
                    
                try:
                    score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
                except ValueError:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_set = U
    
    return best_set

# EVOLVE-BLOCK-START
def construct():
    import itertools
    import math
    import time

    start_time = time.time()
    deadline = 110.0

    best_score = 0
    best_set = [0, 1, 3]

    for n in range(3, 8):
        if time.time() > deadline:
            break
        for max_val in range(n, 50):
            if time.time() > deadline:
                break
            candidates = list(range(1, max_val + 1))
            for combo in itertools.combinations(candidates, n - 1):
                U = [0] + list(combo)
                
                sumset = set()
                diffset = set()
                for a in U:
                    for b in U:
                        sumset.add(a + b)
                        diffset.add(a - b)
                
                size_sum = len(sumset)
                size_diff = len(diffset)
                max_u = max(U)
                
                if size_sum == 0:
                    continue
                    
                ratio = size_diff / size_sum
                if ratio <= 1:
                    continue
                    
                try:
                    score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
                except ValueError:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_set = U
    
    return best_set
# EVOLVE-BLOCK-END
