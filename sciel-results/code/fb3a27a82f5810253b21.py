import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    L = 10**10
    
    powerful_set = set()
    max_v = int(L ** (1/3))
    for v in range(1, max_v + 1):
        v3 = v ** 3
        if v3 > L: break
        max_u = int(math.isqrt(L // v3))
        for u in range(1, max_u + 1):
            powerful_set.add(u * u * v3)
            
    P = sorted(powerful_set)
    n_p = len(P)
    
    counts = {}
    t_deadline = deadline - 10
    
    # Optimization: local variables
    p_list = P
    c_dict = counts
    get = c_dict.get
    time_check_interval = 1024
    last_check = 0
    
    for i in range(n_p):
        a = p_list[i]
        for j in range(i, n_p):
            b = p_list[j]
            s = a + b
            c_dict[s] = get(s, 0) + 1
            
            # Time check
            if j - last_check > time_check_interval:
                if time.time() > t_deadline:
                    break
                last_check = j
        else:
            continue
        break
        
    if not c_dict:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_n, best_count = max(c_dict.items(), key=lambda x: x[1])
    
    pairs = []
    ps = powerful_set
    for a in P:
        b = best_n - a
        if a > b: break
        if b in ps:
            pairs.append([a, b])
            
    return {'N': best_n, 'pairs': pairs}
