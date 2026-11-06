import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    n = 192
    universe = set(range(1, n + 1))
    
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for i, d in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            L = len(others)
            if L < 3:
                continue
            
            # Build residue count map for 'others' mod d
            counts = {}
            for val in others:
                r = val % d
                counts[r] = counts.get(r, 0) + 1
            
            found = False
            for u_idx in range(L):
                u = others[u_idx]
                ru = u % d
                for v_idx in range(u_idx + 1, L):
                    v = others[v_idx]
                    rv = v % d
                    target_r = (- (ru + rv)) % d
                    
                    cnt = counts.get(target_r, 0)
                    if cnt == 0:
                        continue
                    
                    if target_r != ru and target_r != rv:
                        found = True
                        break
                    elif target_r == ru and target_r != rv:
                        if cnt >= 2:
                            found = True
                            break
                    elif target_r != ru and target_r == rv:
                        if cnt >= 2:
                            found = True
                            break
                    elif target_r == ru and target_r == rv:
                        if cnt >= 3:
                            found = True
                            break
            
            if found:
                return False
        
        return True

    def can_add(S, x):
        S_list = list(S)
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S
        if K >= 3:
            for i in range(K):
                u = S_list[i]
                for j in range(i+1, K):
                    v = S_list[j]
                    for k in range(j+1, K):
                        w = S_list[k]
                        if (u + v + w) % x == 0:
                            return False
        
        # Check 2: For each y in S, y divides sum of any triple in S U {x} that includes x
        for y in S_list:
            A = [z for z in S_list if z != y]
            L = len(A)
            if L < 2:
                continue
            for i in range(L):
                u = A[i]
                for j in range(i+1, L):
                    v = A[j]
                    if (x + u + v) % y == 0:
                        return False
        
        return True

    S = set()
    candidates = list(range(n, 0, -1))
    
    for x in candidates:
        if can_add(S, x):
            S.add(x)
        if time.time() - start_time > time_limit * 0.2:
            break
    
    remaining = [x for x in range(1, n + 1) if x not in S]
    random.shuffle(remaining)
    
    for x in remaining:
        if can_add(S, x):
            S.add(x)
        if time.time() - start_time > time_limit * 0.4:
            break
    
    while time.time() - start_time < time_limit:
        improved = False
        
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        if not_in_S:
            random.shuffle(not_in_S)
            for x in not_in_S:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break
        
        if improved:
            continue
        
        if S:
            r = random.choice(list(S))
            if not_in_S:
                x = random.choice(not_in_S)
                new_S = (S - {r}) | {x}
                if is_valid(new_S):
                    S = new_S

    return list(S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
