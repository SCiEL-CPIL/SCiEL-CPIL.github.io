import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def in_S(x, T):
        if x < 1 or x > 128:
            return False
        if x >= 65:
            return True
        return x in T
    
    def valid_T(T):
        T_list = sorted(T)
        n = len(T_list)
        for i in range(n):
            for j in range(i+1, n):
                b, c = T_list[i], T_list[j]
                if b > 63 or c > 63:
                    continue
                s = b + c
                if s >= 65 or s in T:
                    return False
        for i in range(n):
            for j in range(i+1, n):
                a, b = T_list[i], T_list[j]
                ab = a + b
                if not in_S(ab, T):
                    continue
                for k in range(j+1, n):
                    c = T_list[k]
                    if in_S(a+c, T) and in_S(b+c, T):
                        return False
        return True
    
    T = set(range(16, 33))
    best_T = T.copy()
    best_size = len(T)
    
    for it in range(50000):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.4:
            x = random.randint(1, 64)
            if x not in T:
                T.add(x)
                if valid_T(T):
                    if len(T) > best_size:
                        best_size = len(T)
                        best_T = T.copy()
                else:
                    T.discard(x)
        else:
            if T:
                x = random.choice(list(T))
                T.discard(x)
    
    result = list(range(65, 129)) + sorted(best_T)
    return result
