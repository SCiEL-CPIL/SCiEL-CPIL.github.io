import math
import random

def construct():
    N = 512
    
    # 1. Precompute conflicts
    # conflicts[i] is a bitmask of j (1..N) such that lcm(i, j) > N
    # Note: i is 1-indexed. Bit i represents node i.
    conflicts = [0] * (N + 1)
    
    # Helper to compute LCM
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                continue
            if lcm(i, j) > N:
                mask |= (1 << j)
        conflicts[i] = mask

    # Precompute degrees for heuristics
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(conflicts[i]).count('1')

    nodes = list(range(1, N + 1))
    
    best_set = set()
    best_size = 0

    def greedy(order):
        s = set()
        s_mask = 0
        for v in order:
            # Check if v is compatible with all in s
            # v is compatible if no neighbor of v is in s
            # conflicts[v] has bits for neighbors.
            # We need (conflicts[v] & s_mask) == 0
            if (conflicts[v] & s_mask) == 0:
                s.add(v)
                s_mask |= (1 << v)
        return s

    # 1. Try static orders
    orders = [
        nodes,                    # Natural
        list(reversed(nodes)),    # Reverse
        sorted(nodes, key=lambda x: degrees[x]),      # Min degree
        sorted(nodes, key=lambda x: -degrees[x]),     # Max degree
        sorted(nodes, key=lambda x: x),               # Same as natural
        sorted(nodes, key=lambda x: -x),              # Same as reverse
    ]
    
    for order in orders:
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 2. Try random restarts
    # Time budget: 60s.
    # Each greedy pass is ~512 steps.
    # 512 * 10000 = 5e6 ops. Should be < 1s in Python?
    # Python loops are slow. 5e6 iterations might take ~0.5s to 1s.
    # So 50,000 restarts might take ~5s.
    # We have 60s. Let's do 50,000 restarts.
    
    import time
    start_time = time.time()
    max_restarts = 50000
    
    for _ in range(max_restarts):
        if time.time() - start_time > 50: # Leave 10s for safety
            break
            
        random.shuffle(nodes)
        s = greedy(nodes)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 3. Local Search on best_set
    # Try to improve by swapping
    # If we can't find a better one via random restarts, try local search.
    # Actually, random restarts with min-degree heuristic is quite strong.
    
    # Let's try one more heuristic: 
    # Greedy where at each step we pick the available node with min degree 
    # in the *remaining* graph. This is expensive (O(N^2) per pass).
    # N=512, N^2=262k. 512 * 262k = 134M ops. Too slow for many restarts.
    # But we can do a few of these.
    
    def greedy_min_degree_dynamic():
        available = set(range(1, N + 1))
        s = set()
        s_mask = 0
        while available:
            # Find node in available with min conflicts within available
            # conflict_count(v) = popcount(conflicts[v] & mask(available))
            # This is expensive.
            # Optimization: Just pick min degree from precomputed? 
            # The degree changes as we remove nodes.
            # For simplicity and speed, let's stick to static min-degree or random.
            pass
        return s

    return list(best_set)
