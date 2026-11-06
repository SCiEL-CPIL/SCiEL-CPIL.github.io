import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    # Precompute compatibility bitsets
    # can[i] is a bitset of j (1..N) such that lcm(i, j) <= N
    # We use 0-indexed bits: bit j-1 for number j
    can = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                mask |= (1 << (j - 1))
                continue
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << (j - 1))
        can[i] = mask
    
    # Helper to check if x is compatible with set S (represented as bitset)
    def is_compatible(x, S_mask):
        # S_mask has bits set for numbers in S
        # We need to check if all y in S are compatible with x
        # i.e., (can[x] & S_mask) == S_mask
        return (can[x] & S_mask) == S_mask

    # Greedy construction
    def greedy_construct(order):
        S_mask = 0
        S_list = []
        for x in order:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        return S_list

    # Local search
    def local_search(S_list):
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
        
        improved = True
        while improved:
            improved = False
            # Try to add any x
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        S_list.append(x)
                        improved = True
                        break
            
            if not improved:
                # Try to swap: remove one, add one? No, we want to increase size.
                # Try to remove one, add two?
                # This is complex. Let's just try random restarts.
                pass
        
        return S_list

    best_size = 0
    best_set = []
    
    # Try some heuristics
    # 1. Natural order
    s1 = greedy_construct(range(1, N + 1))
    s1 = local_search(s1)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
        
    # 2. Reverse order
    s2 = greedy_construct(range(N, 0, -1))
    s2 = local_search(s2)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
        
    # 3. Random orders
    import random
    for _ in range(1000):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # 4. Try divisors of highly composite numbers
    for M in [360, 420, 480, 240, 120, 72, 60, 48, 36, 24]:
        if time.time() > deadline:
            break
        divs = [i for i in range(1, N + 1) if M % i == 0]
        s = local_search(divs)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return best_set
