import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit

    N = 256

    # Precompute for each x, the list of "conflict pairs" - pairs (a,b) such that 
    # {a,b,x} forms a 3-AP.
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is middle: (x-d, x, x+d)
        for d in range(1, min(x, N - x + 1)):
            a, b = x - d, x + d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is largest: (x-2d, x-d, x)
        for d in range(1, (x + 1) // 2 + 1):
            a, b = x - 2*d, x - d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is smallest: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            a, b = x + d, x + 2*d
            if a <= N and b <= N:
                conflict_pairs[x].append((a, b))

    def can_add(x, in_set):
        """Check if x can be added given current set represented by bool array (1-indexed)."""
        for a, b in conflict_pairs[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def compute_score(S):
        return sum(1.0 / x for x in S)

    # Better initial greedy: sort by 1/x descending (i.e., x ascending), but use a smarter ordering
    # Actually, let's try a few different greedy orderings and pick the best
    
    best_set = set()
    best_score = 0.0
    best_in_set = None
    
    # Try multiple greedy orderings
    orderings = [
        list(range(1, N + 1)),  # natural order
        list(range(1, N + 1, 2)),  # odd numbers first
        sorted(range(1, N + 1), key=lambda x: -1.0/x),  # by reciprocal value
    ]
    
    for ordering in orderings:
        in_set = [False] * (N + 1)
        S = set()
        for x in ordering:
            if can_add(x, in_set):
                in_set[x] = True
                S.add(x)
        
        score = compute_score(S)
        if score > best_score:
            best_score = score
            best_set = S
            best_in_set = in_set[:]
    
    # Local search with multiple strategies
    improved = True
    round_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        round_count += 1
        
        # Strategy 1: Remove-and-readd (remove x, greedily re-fill)
        for x in sorted(best_set):
            if time.time() > deadline:
                break
            
            in_set_temp = best_in_set[:]
            in_set_temp[x] = False
            S_temp = best_set - {x}
            
            new_S = set(S_temp)
            new_in_set = in_set_temp[:]
            
            for y in range(1, N + 1):
                if new_in_set[y]:
                    continue
                if can_add(y, new_in_set):
                    new_in_set[y] = True
                    new_S.add(y)
            
            new_score = compute_score(new_S)
            if new_score > best_score + 1e-15:
                best_set = new_S
                best_score = new_score
                best_in_set = new_in_set
                improved = True
        
        # Strategy 2: Single swap - remove x, add y
        if time.time() < deadline:
            for x in sorted(best_set):
                if time.time() > deadline:
                    break
                
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                best_y = -1
                best_new_score = best_score
                
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        new_score = best_score - 1.0/x + 1.0/y
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_y = y
                
                if best_y > 0 and best_new_score > best_score + 1e-15:
                    best_set = S_temp | {best_y}
                    best_score = best_new_score
                    best_in_set = in_set_temp[:]
                    best_in_set[best_y] = True
                    improved = True
        
        # Strategy 3: 2-swap - remove x, add y, then remove z, add w
        if time.time() < deadline and round_count % 3 == 0:
            elements = sorted(best_set)
            n_elem = len(elements)
            max_x = min(20, n_elem)
            
            for i in range(max_x):
                if time.time() > deadline:
                    break
                x1 = elements[i]
                
                in_set_temp = best_in_set[:]
                in_set_temp[x1] = False
                S_temp = best_set - {x1}
                
                best_y = -1
                best_y_score = best_score
                
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        new_score = best_score - 1.0/x1 + 1.0/y
                        if new_score > best_y_score:
                            best_y_score = new_score
                            best_y = y
                
                if best_y <= 0:
                    continue
                
                in_set_temp2 = in_set_temp[:]
                in_set_temp2[best_y] = True
                S_temp2 = S_temp | {best_y}
                
                for z in sorted(S_temp2):
                    if time.time() > deadline:
                        break
                    if z == best_y:
                        continue
                    
                    in_set_temp3 = in_set_temp2[:]
                    in_set_temp3[z] = False
                    S_temp3 = S_temp2 - {z}
                    
                    best_w = -1
                    best_w_score = best_y_score
                    
                    for w in range(1, N + 1):
                        if in_set_temp3[w]:
                            continue
                        if can_add(w, in_set_temp3):
                            new_score = best_y_score - 1.0/z + 1.0/w
                            if new_score > best_w_score:
                                best_w_score = new_score
                                best_w = w
                    
                    if best_w > 0 and best_w_score > best_score + 1e-15:
                        best_set = S_temp3 | {best_w}
                        best_score = best_w_score
                        best_in_set = in_set_temp3[:]
                        best_in_set[best_w] = True
                        improved = True
                        break
        
        # Strategy 4: Remove two large elements, re-fill greedily
        if time.time() < deadline and round_count % 4 == 0:
            elements = sorted(best_set, reverse=True)
            n_elem = len(elements)
            tried_pairs = set()
            max_pairs_to_try = min(30, n_elem)
            
            for i in range(max_pairs_to_try):
                if time.time() > deadline:
                    break
                for j in range(i + 1, min(i + 15, n_elem)):
                    if time.time() > deadline:
                        break
                    x1, x2 = elements[i], elements[j]
                    key = (x1, x2)
                    if key in tried_pairs:
                        continue
                    tried_pairs.add(key)
                    
                    in_set_temp = best_in_set[:]
                    in_set_temp[x1] = False
                    in_set_temp[x2] = False
                    S_temp = best_set - {x1, x2}
                    
                    new_S = set(S_temp)
                    new_in_set = in_set_temp[:]
                    
                    for y in range(1, N + 1):
                        if new_in_set[y]:
                            continue
                        if can_add(y, new_in_set):
                            new_in_set[y] = True
                            new_S.add(y)
                    
                    new_score = compute_score(new_S)
                    if new_score > best_score + 1e-15:
                        best_set = new_S
                        best_score = new_score
                        best_in_set = new_in_set
                        improved = True

    return list(best_set)
