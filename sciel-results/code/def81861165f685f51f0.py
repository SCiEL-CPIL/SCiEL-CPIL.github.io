import math
from itertools import combinations

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over base-B binary expansion subsets using Greedy + Local Search
    for B in range(2, 6):
        for n in range(4, 8):
            # Generate all base-B binary numbers up to n digits
            candidates = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                candidates.append(val)
            
            # Greedy construction for various target sizes
            for target_size in range(3, 11):
                U = [0]
                # Greedy step
                for _ in range(target_size - 1):
                    best_c = None
                    best_s = -1
                    for c in candidates:
                        if c in U:
                            continue
                        new_U = U + [c]
                        s = calc_score(new_U)
                        if s > best_s:
                            best_s = s
                            best_c = c
                    if best_c is None:
                        break
                    U.append(best_c)
                
                # Local Search (Limited)
                # Try to improve by swapping
                for _pass in range(5):
                    improved = False
                    for i in range(1, len(U)):
                        if improved:
                            break
                        u_i = U[i]
                        for c in candidates:
                            if c in U:
                                continue
                            new_U = U.copy()
                            new_U[i] = c
                            s = calc_score(new_U)
                            if s > best_score:
                                best_score = s
                                best = new_U
                                U = new_U
                                improved = True
                                break
                            elif s > calc_score(U):
                                # Accept if better than current U, even if not global best?
                                # No, we want global best. 
                                # But accepting local improvements helps escape local minima.
                                # Let's just accept if it improves the current U's score.
                                pass 
                    if not improved:
                        break
                
                # Update global best if U is better
                s_final = calc_score(U)
                if s_final > best_score:
                    best_score = s_final
                    best = U

    return best
