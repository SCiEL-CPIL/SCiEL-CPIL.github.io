import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)
    
    # Build adjacency bitmasks
    adj = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i, n):
            b = j + 1
            if (a + b) in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy initial solution
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    used = 0
    solution = set()
    for v in order:
        if (adj[v] & used) == 0:
            used |= (1 << v)
            solution.add(v)
    
    best_solution = set(solution)
    best_size = len(solution)
    
    # Local search: add vertices
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if not (used & (1 << v)) and (adj[v] & used) == 0:
                used |= (1 << v)
                solution.add(v)
                improved = True
                best_size = len(solution)
                best_solution = set(solution)
    
    # Try multiple random orderings
    import random
    for trial in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(trial * 7919 + 1)
        order2 = list(range(n))
        rng.shuffle(order2)
        used2 = 0
        sol2 = set()
        for v in order2:
            if (adj[v] & used2) == 0:
                used2 |= (1 << v)
                sol2.add(v)
        if len(sol2) > best_size:
            best_size = len(sol2)
            best_solution = set(sol2)
            used = used2
            solution = sol2
            # Local search
            improved2 = True
            while improved2 and time.time() < deadline:
                improved2 = False
                for v in range(n):
                    if not (used & (1 << v)) and (adj[v] & used) == 0:
                        used |= (1 << v)
                        solution.add(v)
                        improved2 = True
                        best_size = len(solution)
                        best_solution = set(solution)
    
    # Swap-based local search
    for _ in range(50):
        if time.time() > deadline:
            break
        best_swap = None
        best_gain = 0
        for v in list(solution):
            new_used = used & ~(1 << v)
            new_sol = set(solution)
            new_sol.discard(v)
            added = 0
            for w in range(n):
                if not (new_used & (1 << w)) and (adj[w] & new_used) == 0:
                    new_used |= (1 << w)
                    new_sol.add(w)
                    added += 1
            gain = added - 1
            if gain > best_gain:
                best_gain = gain
                best_swap = (v, new_sol, new_used)
        if best_swap and best_gain > 0:
            v, new_sol, new_used = best_swap
            solution = new_sol
            used = new_used
            best_size = len(solution)
            best_solution = set(solution)
        else:
            break
    
    return sorted(best_solution)
