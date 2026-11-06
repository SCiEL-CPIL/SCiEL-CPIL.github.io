import time
import random

# EVOLVE-BLOCK-START

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute conflict triples
    T = [set() for _ in range(n + 1)]
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            for i in range(4):
                x = ap[i]
                others = tuple(ap[j] for j in range(4) if j != i)
                T[x].add(others)
    
    best_size = 0
    best_set = set()
    
    def can_add(x, S):
        for a, b, c in T[x]:
            if a in S and b in S and c in S:
                return False
        return True
        
    # Heuristic: Sort candidates by "conflict potential" (len(T[x]))
    degrees = [len(T[x]) for x in range(n + 1)]
    
    while time.time() < deadline:
        S = set()
        
        # Mix of strategies
        if random.random() < 0.5:
            candidates = sorted(range(1, n + 1), key=lambda x: degrees[x])
        else:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        
        # Local Search: Add all possible (Single Pass)
        # Since S only grows, we only need one pass
        for x in range(1, n + 1):
            if x not in S and can_add(x, S):
                S.add(x)
        
        # Local Search: Swaps
        # Try to improve by removing one and adding more
        if time.time() < deadline - 0.5:
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                # Limit swap attempts
                for _ in range(20):
                    if not S:
                        break
                    u = random.choice(list(S))
                    S_copy = S.copy()
                    S.remove(u)
                    
                    # Try to add all possible
                    for x in range(1, n + 1):
                        if x not in S and can_add(x, S):
                            S.add(x)
                    
                    if len(S) > len(S_copy):
                        improved = True
                        break
                    else:
                        S = S_copy
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)

# EVOLVE-BLOCK-END
