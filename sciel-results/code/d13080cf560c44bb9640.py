import time
import random
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 140
    
    # Fallback result
    squares = {x * x % 17 for x in range(1, 17)}
    fallback = {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    # Helper functions
    def has_K4(A, n):
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    common = []
                    for k in range(n):
                        if k != i and k != j and A[i, k] == 1 and A[j, k] == 1:
                            common.append(k)
                    for idx, u in enumerate(common):
                        for v in common[idx + 1:]:
                            if A[u, v] == 1:
                                return True
        return False

    def has_ind_set_6(A, n):
        vertices = list(range(n))
        degrees = [int(np.sum(A[i])) for i in range(n)]
        vertices.sort(key=lambda x: degrees[x])
        
        def backtrack(start, current_set):
            if len(current_set) == 6:
                return True
            if len(current_set) + (n - start) < 6:
                return False
            
            for i in range(start, n):
                v = vertices[i]
                ok = True
                for u in current_set:
                    if A[u, v] == 1:
                        ok = False
                        break
                if ok:
                    if backtrack(i + 1, current_set + [v]):
                        return True
            return False
            
        return backtrack(0, [])

    def get_cost(A, n):
        c1 = 1 if has_K4(A, n) else 0
        c2 = 1 if has_ind_set_6(A, n) else 0
        return c1 + c2

    def solve_n(n, time_limit):
        end_time = time.time() + time_limit
        A = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        
        current_cost = get_cost(A, n)
        if current_cost == 0:
            return A
        
        best_A = A.copy()
        best_cost = current_cost
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() < end_time:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            A[u, v] = 1 - A[u, v]
            A[v, u] = 1 - A[v, u]
            
            new_cost = get_cost(A, n)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = A.copy()
                if new_cost == 0:
                    return A
            
            if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / T):
                current_cost = new_cost
            else:
                A[u, v] = 1 - A[u, v]
                A[v, u] = 1 - A[v, u]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_cost == 0:
            return best_A
        return None

    # Try n from 35 down to 18
    # Allocate time dynamically
    # 35: 40s, 34: 20s, 33: 15s, 32: 10s, 31: 5s, 30: 5s, ...
    time_alloc = {
        35: 40,
        34: 20,
        33: 15,
        32: 10,
        31: 5,
        30: 5,
        29: 5,
        28: 5,
        27: 5,
        26: 5,
        25: 5,
        24: 5,
        23: 5,
        22: 5,
        21: 5,
        20: 5,
        19: 5,
        18: 5
    }
    
    for n in range(35, 17, -1):
        if time.time() > total_deadline - 5:
            break
        limit = time_alloc.get(n, 5)
        # Cap limit to remaining time
        remaining = total_deadline - time.time() - 5
        if limit > remaining:
            limit = remaining
        if limit <= 0:
            break
            
        result_A = solve_n(n, limit)
        if result_A is not None:
            return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if result_A[i, j] == 1]}
            
    return fallback

