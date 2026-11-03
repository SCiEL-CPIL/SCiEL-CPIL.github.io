import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other params
        # Simple greedy for general case
        M = q**n
        balls = []
        for i in range(M):
            ball = set([i])
            digits = [0]*n
            temp = i
            for j in range(n):
                digits[j] = temp % q
                temp //= q
            for pos in range(n):
                for d in range(q):
                    if d == digits[pos]: continue
                    new_digits = digits[:]
                    new_digits[pos] = d
                    idx = 0
                    mult = 1
                    for j in range(n):
                        idx += new_digits[j] * mult
                        mult *= q
                    ball.add(idx)
            balls.append(list(ball))
        
        code = set()
        uncovered = set(range(M))
        while uncovered:
            best_c = None
            max_cover = -1
            for c in range(M):
                if c in code: continue
                cover = len(uncovered.intersection(balls[c]))
                if cover > max_cover:
                    max_cover = cover
                    best_c = c
            if best_c is None: break
            code.add(best_c)
            uncovered -= set(balls[best_c])
            
        return [list(map(int, [(c // (q**i)) % q for i in range(n)])) for c in code]
    
    M = 3**n
    ball_size = 1 + 2*n # 17
    
    # Precompute balls
    balls = [None] * M
    for i in range(M):
        ball = [i]
        digits = [0]*n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        for pos in range(n):
            p_val = 3**pos
            # +1
            new_digit = (digits[pos] + 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                ball.append(i + diff * p_val)
            # -1
            new_digit = (digits[pos] - 1) % 3
            diff = new_digit - digits[pos]
            if diff != 0:
                ball.append(i + diff * p_val)
        balls[i] = ball
        
    # Precompute coverers
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    # Helper to calculate coverage count
    def get_count(code_set):
        count = np.zeros(M, dtype=np.int32)
        for c in code_set:
            for p in balls[c]:
                count[p] += 1
        return count

    # Helper: Local Search (Reduce size)
    def local_search(code_set, time_limit):
        count = get_count(code_set)
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            
            # Phase 1: Remove redundant
            to_remove = []
            for c in list(code_set):
                redundant = True
                for p in balls[c]:
                    if count[p] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
                    
            if to_remove:
                for c in to_remove:
                    code_set.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                improved = True
                continue
            
            # Phase 2: 1-Swap
            code_list = list(code_set)
            random.shuffle(code_list)
            
            swapped = False
            for c in code_list:
                if time.time() >= time_limit:
                    break
                unique_points = [p for p in balls[c] if count[p] == 1]
                
                if not unique_points:
                    code_set.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    improved = True
                    swapped = True
                    break
                
                # Find candidates covering all unique_points
                candidates = set(coverers[unique_points[0]])
                for p in unique_points[1:]:
                    candidates.intersection_update(coverers[p])
                
                candidates -= code_set
                
                if candidates:
                    # Heuristic: pick candidate that covers most 'soft' points (count > 1)
                    # to potentially enable future removals?
                    # Or just pick one that minimizes "cost" if we had weights.
                    # Let's pick the one with highest "redundancy" potential?
                    # Actually, any swap keeps size constant.
                    # To escape local minima, random selection is good.
                    c_new = random.choice(list(candidates))
                    
                    code_set.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    code_set.add(c_new)
                    for p in balls[c_new]:
                        count[p] += 1
                    
                    improved = True
                    swapped = True
                    break
            
            if not swapped:
                break
                
        return code_set, count

    # Main ILS Loop
    start_time = time.time()
    time_limit = 118.0
    
    # Initial Greedy
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.full(M, ball_size, dtype=np.int32)
    heap = [(-ball_size, c) for c in range(M)]
    heapq.heapify(heap)
    uncovered_count = M
    
    while uncovered_count > 0 and time.time() < time_limit:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
            valid_c = c
            break
        if valid_c is None:
            break
        code.add(valid_c)
        for p in balls[valid_c]:
            if uncovered[p]:
                uncovered[p] = False
                uncovered_count -= 1
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        heapq.heappush(heap, (-gains[c_prime], c_prime))
                        
    best_code = set(code)
    best_size = len(code)
    
    # Run local search on initial greedy
    code, count = local_search(code, time_limit)
    if len(code) < best_size:
        best_code = set(code)
        best_size = len(code)
        
    # ILS Loop
    while time.time() < time_limit - 5.0: # Leave 5s for final return processing
        # Perturbation
        # Remove k random codewords
        k = random.randint(3, 8)
        if len(code) <= k:
            # Reset to best
            code = set(best_code)
            count = get_count(code)
            continue
            
        to_remove = random.sample(list(code), k)
        for c in to_remove:
            code.remove(c)
            for p in balls[c]:
                count[p] -= 1
        
        # Repair: Greedily cover uncovered
        # Find uncovered points
        uncovered_indices = np.where(count == 0)[0]
        if len(uncovered_indices) > 0:
            # Greedy repair
            # We can just run a mini-greedy on the uncovered set
            # To be fast, we can just pick random codewords that cover uncovered points?
            # Better: Greedy max coverage on uncovered.
            uncov_set = set(uncovered_indices)
            while uncov_set:
                # Find best codeword covering most uncov_set
                # This is O(M * |uncov_set|) which might be slow if done repeatedly.
                # Approximation: Pick a random uncovered point, pick a random codeword covering it?
                # This is a "Randomized Greedy".
                
                # To keep it fast, let's just add codewords that cover the most currently uncovered points.
                # Since uncov_set is small (usually < 50), we can iterate over all M? No, M=6561.
                # 6561 * 50 = 300k ops. Fast.
                
                best_c = None
                max_gain = 0
                # Optimization: Only consider codewords that cover at least one uncovered point?
                # We can precompute for each uncovered point, its coverers.
                # Union of coverers of uncovered points.
                
                candidates = set()
                for p in uncov_set:
                    candidates.update(coverers[p])
                
                if not candidates:
                    break # Should not happen if coverers is correct
                    
                for c in candidates:
                    if c in code:
                        continue
                    gain = 0
                    for p in balls[c]:
                        if p in uncov_set:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_c = c
                        
                if best_c is None:
                    # Fallback: add random
                    best_c = random.choice(list(candidates))
                
                code.add(best_c)
                for p in balls[best_c]:
                    count[p] += 1
                    if p in uncov_set and count[p] > 0:
                        uncov_set.remove(p)
                        
        # Local Search on the repaired code
        code, count = local_search(code, time_limit)
        
        if len(code) < best_size:
            best_code = set(code)
            best_size = len(code)
            
        # Optional: If we hit the lower bound or time is tight, break
        if best_size <= 403:
            break

    # Return best_code
    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in best_code]
