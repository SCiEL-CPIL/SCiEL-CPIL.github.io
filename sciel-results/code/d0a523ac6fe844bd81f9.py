import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 512
    
    def can_add(S, x):
        for d in range(1, (x-1)//3 + 1):
            if S[x-3*d] and S[x-2*d] and S[x-d]:
                return False
        limit = min((x-1)//2, n-x)
        for d in range(1, limit + 1):
            if S[x-2*d] and S[x-d] and S[x+d]:
                return False
        limit = min(x-1, (n-x)//2)
        for d in range(1, limit + 1):
            if S[x-d] and S[x+d] and S[x+2*d]:
                return False
        for d in range(1, (n-x)//3 + 1):
            if S[x+d] and S[x+2*d] and S[x+3*d]:
                return False
        return True

    best_size = 0
    best_set = []
    
    while time.time() < deadline:
        S = bytearray(n + 1)
        size = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S[x] = 1
                size += 1
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            for x in range(1, n + 1):
                if not S[x] and can_add(S, x):
                    S[x] = 1
                    size += 1
                    improved = True
                    break
            
            # Try swap
            if not improved:
                for _ in range(5):
                    if time.time() >= deadline: break
                    y = random.randint(1, n)
                    if S[y]:
                        S_temp = bytearray(S)
                        S_temp[y] = 0
                        size_temp = size - 1
                        
                        # Greedily add to S_temp
                        candidates = list(range(1, n + 1))
                        random.shuffle(candidates)
                        for x in candidates:
                            if not S_temp[x] and can_add(S_temp, x):
                                S_temp[x] = 1
                                size_temp += 1
                                
                        if size_temp > size:
                            S = S_temp
                            size = size_temp
                            improved = True
                            break
                            
        if size > best_size:
            best_size = size
            best_set = [i for i in range(1, n+1) if S[i]]

    return best_set
