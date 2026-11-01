import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Precompute base for index <-> word
    # Not strictly needed if we just use indices, but needed for output
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: list of lists of indices
    # ball_indices[i] is a list of 15 indices
    ball_indices = [None] * total
    for i in range(total):
        temp = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        
        ball = [i]
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                ball.append(i + (v - orig) * base[coord])
        ball_indices[i] = ball

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        # Optimize greedy: 
        # Instead of scanning all 2187 every time (2187 * 15 ops * 200 iters ~ 6.5M ops),
        # we can just do it. It's fast enough.
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            # Randomize start to break symmetry
            start = random.randint(0, total - 1)
            # We check all candidates. 
            # To speed up, we can skip candidates that are already covered? 
            # No, a covered candidate might cover *other* uncovered points.
            
            # Optimization: Only check candidates that cover at least 1 uncovered point?
            # Actually, if a candidate's ball is fully covered, new_cov is 0.
            # We can precompute which candidates are "active".
            
            # Simple loop
            # Use a list of all indices and shuffle? 
            # Shuffling 2187 items is fast.
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices[c]
                # Count uncovered in ball
                # covered is a numpy array. 
                # new_cov = sum(covered[x] == 0 for x in ball)
                # Using numpy:
                new_cov = np.count_nonzero(covered[ball] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            code.append(best_c)
            covered[ball_indices[best_c]] = 1
            uncovered_count = np.count_nonzero(covered == 0)
            
        return code

    # Local Search
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    # Check if c is redundant
                    # min(cnt[ball_indices[c]]) >= 2
                    if np.min(cnt[ball_indices[c]]) >= 2:
                        code_set.remove(c)
                        cnt[ball_indices[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                            if best_size <= 160: # Early exit if we hit lower bound vicinity? No, keep going.
                                pass

        # Initial delete pass
        try_delete()

        # Swap loop
        # We try to swap a codeword with a non-codeword if it maintains validity
        # Validity: For all x in B(u) \ B(v), cnt[x] >= 2 (before removing u)
        
        no_success_streak = 0
        max_streak = 1000
        
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break
            
            # Pick a codeword to remove
            c_remove = random.choice(list(code_set))
            # Pick a codeword to add
            c_add = random.randint(0, total - 1)
            
            if c_add in code_set or c_add == c_remove:
                continue
                
            ball_rem = ball_indices[c_remove]
            ball_add = ball_indices[c_add]
            
            # Check validity
            # We need to ensure that any point covered ONLY by c_remove is also covered by c_add
            # i.e., if x in ball_rem and x not in ball_add, then cnt[x] must be >= 2 (so another codeword covers it)
            
            valid = True
            # Optimize check: 
            # Create a set of ball_add for O(1) lookup? 
            # Ball size is 15. List check is O(15). 15*15 = 225 ops. 
            # Set check: 15 inserts + 15 lookups. Faster.
            add_set = set(ball_add)
            
            for x in ball_rem:
                if x not in add_set:
                    if cnt[x] < 2:
                        valid = False
                        break
            
            if valid:
                # Perform swap
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[ball_rem] -= 1
                cnt[ball_add] += 1
                no_success_streak = 0
                
                # Try to delete redundant words after swap
                try_delete()
            else:
                no_success_streak += 1
                if no_success_streak > max_streak:
                    # If stuck, restart with a different greedy seed? 
                    # Or just continue. 
                    # Let's just continue, randomness should eventually find a path.
                    pass

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=10.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 5.0 and best_size_overall > 160:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 5.0 - time.time()
        # Allocate more time to LS if we are getting good results?
        # Or just fixed budget per restart.
        tb = min(8.0, max(1.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final conversion
    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
