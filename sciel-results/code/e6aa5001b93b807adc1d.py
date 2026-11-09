import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for evaluator's 60s limit
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check compatibility
    def is_compatible(x, S, sums):
        if 2 * x in sums:
            return False
        for a in S:
            if x + a in sums:
                return False
        return True

    def get_candidates(S, sums):
        C = []
        for x in range(1, N + 1):
            if x in S:
                continue
            if is_compatible(x, S, sums):
                C.append(x)
        return C

    def greedy_blocking_restart(seed):
        random.seed(seed)
        S = []
        sums = set()
        
        while True:
            C = get_candidates(S, sums)
            if not C:
                break
            
            # Calculate blocking score
            # x blocks y if x + y in sums
            scores = {}
            for x in C:
                count = 0
                for y in C:
                    if x == y:
                        continue
                    if x + y in sums:
                        count += 1
                scores[x] = count
            
            max_score = max(scores.values())
            candidates_max = [x for x in C if scores[x] == max_score]
            x_star = random.choice(candidates_max)
            
            # Add x_star
            S.append(x_star)
            sums.add(2 * x_star)
            for a in S[:-1]:
                sums.add(x_star + a)
                
            if time.time() > deadline:
                break
        
        return S

    def greedy_random_order(seed):
        random.seed(seed)
        S = []
        sums = set()
        available = list(range(1, N + 1))
        random.shuffle(available)
        
        for x in available:
            if x in S:
                continue
            if is_compatible(x, S, sums):
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S

    # Run blocking restarts
    # We estimate 1 restart takes ~1ms. 55s allows ~50k restarts.
    # We will loop until deadline.
    seed = 0
    while time.time() < deadline:
        S = greedy_blocking_restart(seed)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
        seed += 1
        
        # Intersperse random order restarts
        if seed % 10 == 0:
            S2 = greedy_random_order(seed)
            if len(S2) < best_size:
                best_size = len(S2)
                best_set = S2

    if best_set is None:
        # Fallback
        S = []
        sums = set()
        for x in range(1, N + 1):
            if is_compatible(x, S, sums):
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        best_set = S

    return best_set
