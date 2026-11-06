import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 256
    
    def is_valid_with(S, x):
        """
        Check if x can be added to set S.
        S is a set of ints.
        """
        S_list = list(S)
        # Case 1: x is a
        # x < b <= c in S
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b
        # a < x <= c in S
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
            
        # Case 3: x is c
        # a < b <= x in S
        for a in S_list:
            if a >= x:
                continue
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_with(S, x):
                S.add(x)
        
        # Polish: try to add any remaining elements
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in S:
                    if is_valid_with(S, x):
                        S.add(x)
                        changed = True
        return S

    best_S = set()
    
    # 1. Descending order
    order_desc = list(range(n, 0, -1))
    S_desc = greedy(order_desc)
    if len(S_desc) > len(best_S):
        best_S = S_desc
        
    # 2. Ascending order (unlikely to be best, but check)
    order_asc = list(range(1, n + 1))
    S_asc = greedy(order_asc)
    if len(S_asc) > len(best_S):
        best_S = S_asc
        
    # 3. Random orders
    # Limit number of restarts to stay within time
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        random.seed(i)
        order_rand = list(range(1, n + 1))
        random.shuffle(order_rand)
        S_rand = greedy(order_rand)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    return sorted(best_S)
