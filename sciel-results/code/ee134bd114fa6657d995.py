import itertools
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    # Specific case for q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        N = q ** n
        words = [list(t) for t in itertools.product(range(q), repeat=n)]
        word_to_idx = {tuple(w): i for i, w in enumerate(words)}
        
        # Precompute balls
        balls = [[] for _ in range(N)]
        for j in range(N):
            w = words[j]
            balls[j].append(j)
            for pos in range(n):
                for delta in [1, -1]:
                    new_val = (w[pos] + delta) % q
                    if new_val != w[pos]:
                        new_w = w.copy()
                        new_w[pos] = new_val
                        idx = word_to_idx[tuple(new_w)]
                        balls[j].append(idx)
                        
        # ILP Setup
        c = np.ones(N)
        A = np.zeros((N, N), dtype=np.float64)
        for j in range(N):
            for i in balls[j]:
                A[i, j] = 1.0
                
        cons = LinearConstraint(A, 1.0, np.inf)
        bounds = Bounds(0, 1)
        integrality = np.ones(N)
        
        try:
            options = {"time_limit": 100}
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options=options)
            
            # Check result
            if res.x is not None:
                x = res.x
                # If optimal or feasible
                if res.status in [0, 3]: # 0: Optimal, 3: Time Limit (but feasible found?)
                    # For status 3, x might be the best feasible found.
                    # We must verify feasibility because HiGHS might return fractional if not strictly integer?
                    # No, integrality is set. But if time limit, it returns the best incumbent.
                    # Round to be safe?
                    x_int = np.round(x).astype(int)
                    
                    # Verify coverage
                    cover = np.zeros(N, dtype=np.int32)
                    for j in range(N):
                        if x_int[j] == 1:
                            for i in balls[j]:
                                cover[i] += 1
                                
                    if np.min(cover) >= 1:
                        code = [i for i in range(N) if x_int[i] == 1]
                        return [words[i] for i in code]
                        
        except Exception:
            pass
            
        # Fallback to greedy if ILP failed
        return _greedy_construct(q, n, radius)
        
    else:
        return _greedy_construct(q, n, radius)

def _greedy_construct(q, n, radius):
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute distances? No, just check on fly for simplicity in fallback
    # This is slow but only used for non-standard cases
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            w = words[i]
            cov = 0
            for u_idx in uncovered:
                u = words[u_idx]
                dist = 0
                for k in range(n):
                    if w[k] != u[k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        w = words[best_i]
        to_remove = []
        for u_idx in uncovered:
            u = words[u_idx]
            dist = 0
            for k in range(n):
                if w[k] != u[k]:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                to_remove.append(u_idx)
        uncovered.difference_update(to_remove)
        
    return [words[i] for i in code]
