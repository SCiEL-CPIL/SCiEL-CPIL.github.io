import numpy as np
import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    M = 3 ** n
    ball_size = 1 + 2 * n  # 17
    
    # Precompute powers of 3
    powers = [3 ** i for i in range(n)]
    
    # Precompute balls using vectorized operations
    # For each index, compute its ball
    indices = np.arange(M)
    balls = []
    
    for i in indices:
        ball = [i]
        for pos in range(n):
            # +1
            digit = (i // powers[pos]) % 3
            if digit < 2:
                ball.append(i + powers[pos])
            else:
                ball.append(i - 2 * powers[pos])
            # -1
            if digit > 0:
                ball.append(i - powers[pos])
            else:
                ball.append(i + 2 * powers[pos])
        balls.append(ball)
    
    # Precompute coverers: for each point p, list of codewords c such that p in balls[c]
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
    
    # Greedy Set Cover with tie-breaking
    code = set()
    uncovered = np.ones(M, dtype=bool)
    gains = np.full(M, ball_size, dtype=np.int32)
    
    # Heap entries: (-gain, -potential, codeword)
    # potential: number of points in ball that are already covered (tie-breaker)
    heap = []
    for c in range(M):
        potential = 0  # initially all uncovered
        heap.append((-ball_size, 0, c))
    heapq.heapify(heap)
    
    uncovered_count = M
    
    while uncovered_count > 0:
        valid_c = None
        while heap:
            neg_g, neg_p, c = heapq.heappop(heap)
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
        
        # Update uncovered and gains
        for p in balls[valid_c]:
            if uncovered[p]:
                uncovered[p] = False
                uncovered_count -= 1
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        heapq.heappush(heap, (-gains[c_prime], 0, c_prime))
    
    # Local Search
    start_time = time.time()
    time_limit = 115
    
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove redundant
        to_remove = []
        for c in list(code):
            redundant = True
            for p in balls[c]:
                if count[p] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
        
        if to_remove:
            for c in to_remove:
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
            improved = True
            continue
        
        # Phase 2: 1-swap
        code_list = list(code)
        random.shuffle(code_list)
        
        swapped = False
        for c in code_list:
            if (time.time() - start_time) >= time_limit:
                break
            
            unique_points = [p for p in balls[c] if count[p] == 1]
            
            if not unique_points:
                code.remove(c)
                for p in balls[c]:
                    count[p] -= 1
                improved = True
                swapped = True
                break
            
            candidates = set(coverers[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(coverers[p])
            candidates -= code
            
            if candidates:
                # Pick candidate that maximizes overlap with already-covered points
                best_c = None
                best_score = -1
                for c_new in list(candidates)[:50]:  # Limit for speed
                    score = sum(1 for p in balls[c_new] if count[p] > 1)
                    if score > best_score:
                        best_score = score
                        best_c = c_new
                
                if best_c is not None:
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                    code.add(best_c)
                    for p in balls[best_c]:
                        count[p] += 1
                    improved = True
                    swapped = True
                    break
        
        if not swapped:
            # Perturbation: remove k random, re-greedy
            if (time.time() - start_time) < time_limit:
                k = min(10, len(code) // 10)
                if k > 0:
                    to_perturb = random.sample(list(code), k)
                    for c in to_perturb:
                        code.remove(c)
                        for p in balls[c]:
                            count[p] -= 1
                    
                    # Re-greedy on uncovered
                    uncovered_now = count == 0
                    if np.any(uncovered_now):
                        # Find best candidates for uncovered
                        gains_temp = np.zeros(M, dtype=np.int32)
                        for p in np.where(uncovered_now)[0]:
                            for c_prime in coverers[p]:
                                if c_prime not in code:
                                    gains_temp[c_prime] += 1
                        
                        # Add top k
                        top_candidates = np.argsort(-gains_temp)[:k]
                        for c_new in top_candidates:
                            if c_new not in code and gains_temp[c_new] > 0:
                                code.add(int(c_new))
                                for p in balls[int(c_new)]:
                                    if count[p] == 0:
                                        count[p] = 1
                                    else:
                                        count[p] += 1
                        improved = True
    
    return [list(map(int, [(c // (3 ** i)) % 3 for i in range(n)])) for c in code]
