import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    S = []
    used_sums = set()
    
    max_val = 4096
    
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
            
        # Generate all new triple sums involving x
        new_sums_list = []
        
        # 1. (x, x, x)
        new_sums_list.append(3 * x)
        
        # 2. (x, x, a) for a in S
        for a in S:
            new_sums_list.append(2 * x + a)
            
        # 3. (x, a, b) for a <= b in S
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                new_sums_list.append(x + a + b)
                
        # Check for internal collisions
        if len(new_sums_list) != len(set(new_sums_list)):
            continue
            
        # Check for collisions with existing sums
        new_sums_set = set(new_sums_list)
        if new_sums_set & used_sums:
            continue
            
        # Valid candidate
        S.append(x)
        used_sums.update(new_sums_set)
        
    return S
