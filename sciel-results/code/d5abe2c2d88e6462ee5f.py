import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 512
    # adj[i] is bitmask of j compatible with i
    adj = [0] * n
    for i in range(n):
        num_i = i + 1
        bits = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= 512:
                bits |= (1 << j)
        adj[i] = bits

    best_size = 0
    best_clique = []
    
    def is_compatible(x, y):
        # x, y are indices 0..511
        return (adj[x] >> y) & 1
    
    def greedy_clique(order):
        clique = []
        for v in order:
            ok = True
            for u in clique:
                if not is_compatible(v, u):
                    ok = False
                    break
            if ok:
                clique.append(v)
        return clique

    # Initial greedy with natural order
    current_clique = greedy_clique(list(range(n)))
    best_size = len(current_clique)
    best_clique = current_clique[:]

    # Local Search: Try to add a vertex by removing one
    # Or swap
    
    # Run restarts
    while time.time() < deadline:
        # Random shuffle
        order = list(range(n))
        random.shuffle(order)
        
        # To bias towards good vertices, we can sort by degree desc after shuffle?
        # Or just shuffle.
        
        c = greedy_clique(order)
        
        # Local search on c
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any vertex not in c
            c_set = set(c)
            for v in range(n):
                if v in c_set:
                    continue
                # Check if v is compatible with all in c
                ok = True
                for u in c:
                    if not is_compatible(v, u):
                        ok = False
                        break
                if ok:
                    c.append(v)
                    c_set.add(v)
                    improved = True
                    break # Restart loop
            
            # If no addition possible, try swap
            if not improved:
                for i in range(len(c)):
                    u = c[i]
                    for v in range(n):
                        if v in c_set:
                            continue
                        # Check if v is compatible with c \ {u}
                        ok = True
                        for k in range(len(c)):
                            if k == i:
                                continue
                            if not is_compatible(v, c[k]):
                                ok = False
                                break
                        if ok:
                            c[i] = v
                            c_set.remove(u)
                            c_set.add(v)
                            improved = True
                            break
                    if improved:
                        break

        if len(c) > best_size:
            best_size = len(c)
            best_clique = c[:]
            
    # Convert indices to values
    return [v + 1 for v in best_clique]
