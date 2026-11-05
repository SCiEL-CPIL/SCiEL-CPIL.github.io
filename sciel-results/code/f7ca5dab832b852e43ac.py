import numpy as np
import random

def _precompute_balls_and_coverers(q, n, radius):
    N = q ** n
    balls = []
    for i in range(N):
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
        
        covered = set([i])
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                weight = q ** (n - 1 - pos)
                for new_d in range(q):
                    if new_d == orig:
                        continue
                    delta = (new_d - orig) * weight
                    ni = i + delta
                    covered.add(ni)
        balls.append(covered)
        
    coverers = [set() for _ in range(N)]
    for i in range(N):
        for w in balls[i]:
            coverers[w].add(i)
            
    return balls, coverers

def _greedy_cover_local_search(q, n, radius, num_restarts=20):
    N = q ** n
    balls, coverers = _precompute_balls_and_coverers(q, n, radius)
    ball_lists = [list(b) for b in balls] # For iteration
    
    best_code = None
    best_size = float('inf')
    
    for _ in range(num_restarts):
        # 1. Greedy Initialization
        uncovered = [True] * N
        counts = [len(b) for b in balls]
        selected = set()
        
        while any(uncovered):
            max_c = max(counts)
            if max_c == 0:
                break
            candidates = [i for i in range(N) if counts[i] == max_c]
            chosen = random.choice(candidates)
            selected.add(chosen)
            
            for w in ball_lists[chosen]:
                if uncovered[w]:
                    uncovered[w] = False
                    for i in coverers[w]:
                        counts[i] -= 1
                        
        # 2. Local Search
        cov_counts = [0] * N
        for c in selected:
            for w in ball_lists[c]:
                cov_counts[w] += 1
                
        improved = True
        while improved:
            improved = False
            
            # Try deletions
            for c in list(selected):
                can_remove = True
                for w in ball_lists[c]:
                    if cov_counts[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected.remove(c)
                    for w in ball_lists[c]:
                        cov_counts[w] -= 1
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Try swaps
            # Limit number of swap attempts per iteration to prevent infinite loops in plateaus
            swaps_made = 0
            while swaps_made < 50:
                c1 = random.choice(list(selected))
                c2 = random.randint(0, N - 1)
                if c2 in selected:
                    continue
                    
                valid = True
                for w in ball_lists[c1]:
                    if cov_counts[w] == 1:
                        if w not in balls[c2]:
                            valid = False
                            break
                            
                if valid:
                    selected.remove(c1)
                    selected.add(c2)
                    for w in ball_lists[c1]:
                        cov_counts[w] -= 1
                    for w in ball_lists[c2]:
                        cov_counts[w] += 1
                    improved = True
                    swaps_made += 1
                    break # Restart outer loop to check deletions again
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = list(selected)
            
    result = []
    if best_code:
        for idx in best_code:
            digits = []
            temp = idx
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            digits.reverse()
            result.append(digits)
    return result

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _greedy_cover_local_search(3, 6, 1, num_restarts=50)
    else:
        return _greedy_cover_local_search(q, n, radius, num_restarts=10)
