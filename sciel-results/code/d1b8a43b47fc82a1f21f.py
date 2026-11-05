import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit
    
    n = 24
    rng = random.Random(614)
    
    # Generate the shuffled edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    
    # Map edge to its 1-based position
    pos = {}
    for i, (u, v) in enumerate(edges):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1
        
    def edge_cost(u, v):
        return pos[(u, v)]
        
    def max_cost(cyc):
        m = 0
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            c = edge_cost(u, v)
            if c > m:
                m = c
        return m
        
    best_cycle = list(range(n))
    best_score = max_cost(best_cycle)
    
    # Try multiple restarts
    # Start with identity, then random permutations
    rng_search = random.Random(123)
    
    for restart in range(100): # Up to 100 restarts
        if time.time() > deadline:
            break
            
        if restart == 0:
            cycle = list(range(n))
        else:
            cycle = list(range(n))
            rng_search.shuffle(cycle)
            
        current_max = max_cost(cycle)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_move = None
            best_new_max = current_max
            
            for i in range(n):
                u1 = cycle[i]
                v1 = cycle[(i + 1) % n]
                
                for j in range(i + 1, n):
                    if j == i + 1:
                        continue
                    if i == 0 and j == n - 1:
                        continue
                        
                    u2 = cycle[j]
                    v2 = cycle[(j + 1) % n]
                    
                    e1 = (u1, u2)
                    e2 = (v1, v2)
                    nc1 = edge_cost(e1[0], e1[1])
                    nc2 = edge_cost(e2[0], e2[1])
                    
                    # Calculate max of remaining edges
                    max_remaining = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        u = cycle[k]
                        v = cycle[(k + 1) % n]
                        c = edge_cost(u, v)
                        if c > max_remaining:
                            max_remaining = c
                            
                    new_max = max(max_remaining, nc1, nc2)
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (i, j)
                        
            if best_move is not None:
                i, j = best_move
                cycle[i+1:j+1] = cycle[i+1:j+1][::-1]
                current_max = best_new_max
                improved = True
            else:
                # Local minimum reached
                break
                
        if current_max < best_score:
            best_score = current_max
            best_cycle = cycle[:]
            
    return best_cycle

if __name__ == "__main__":
    print(construct())
