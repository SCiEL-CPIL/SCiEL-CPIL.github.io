import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # safety margin before 120s kill
    
    def greedy_fill(adj, common, seed):
        rng = random.Random(seed)
        for _ in range(5000):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or v in adj[u]:
                continue
            if common[u][v] > 2:
                continue
            ok = True
            for y in adj[v]:
                if common[u][y] > 1:
                    ok = False
                    break
            if not ok:
                continue
            for y in adj[u]:
                if common[v][y] > 1:
                    ok = False
                    break
            if ok:
                adj[u].add(v)
                adj[v].add(u)
                for y in adj[v]:
                    common[u][y] += 1
                    common[y][u] += 1
                for y in adj[u]:
                    common[v][y] += 1
                    common[y][v] += 1
    
    def can_add(adj, common, u, v):
        if u == v or v in adj[u]:
            return False
        if common[u][v] > 2:
            return False
        for y in adj[v]:
            if common[u][y] > 1:
                return False
        for y in adj[u]:
            if common[v][y] > 1:
                return False
        return True
    
    def add_edge(adj, common, u, v):
        adj[u].add(v)
        adj[v].add(u)
        for y in adj[v]:
            common[u][y] += 1
            common[y][u] += 1
        for y in adj[u]:
            common[v][y] += 1
            common[y][v] += 1
    
    def remove_edge(adj, common, u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        for y in adj[v]:
            common[u][y] -= 1
            common[y][u] -= 1
        for y in adj[u]:
            common[v][y] -= 1
            common[y][v] -= 1
    
    def get_edge_count(adj):
        return sum(len(s) for s in adj) // 2
    
    def greedy_polish(adj, common, rng, iterations=200):
        for _ in range(iterations):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or v in adj[u]:
                continue
            if can_add(adj, common, u, v):
                add_edge(adj, common, u, v)
    
    def simulated_annealing(adj, common, rng, T_start, T_end, time_budget):
        T = T_start
        current_count = get_edge_count(adj)
        sa_end_time = start_time + time_budget
        
        while start_time and (time.time() - start_time) < sa_end_time:
            if T < T_end:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add an edge
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or v in adj[u]:
                    continue
                if can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_count += 1
                # If can't add, skip (no state change)
                
            elif move_type < 0.8:
                # Try to remove an edge
                if current_count == 0:
                    continue
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if not edges_list:
                    continue
                u, v = rng.choice(edges_list)
                remove_edge(adj, common, u, v)
                new_count = current_count - 1
                delta = -1
                if rng.random() < math.exp(delta / T):
                    current_count = new_count
                else:
                    add_edge(adj, common, u, v)
                    
            else:
                # 2-opt swap: remove one edge, try to add another
                if current_count == 0:
                    continue
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if not edges_list:
                    continue
                u1, v1 = rng.choice(edges_list)
                
                # Try several candidate edges to add
                best_swap = None
                for _ in range(10):
                    u2 = rng.randint(0, n - 1)
                    v2 = rng.randint(0, n - 1)
                    if u2 == v2 or v2 in adj[u2]:
                        continue
                    if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                        continue
                    # Temporarily remove u1-v1 and check if we can add u2-v2
                    remove_edge(adj, common, u1, v1)
                    if can_add(adj, common, u2, v2):
                        add_edge(adj, common, u2, v2)
                        best_swap = (u2, v2)
                        break
                    else:
                        add_edge(adj, common, u1, v1)
                
                if best_swap:
                    # We already removed u1-v1 and added u2-v2 in the loop
                    # current_count stays the same (swap)
                    pass
                else:
                    # No valid swap found, state unchanged
                    pass

            T *= 0.9995
        
        return current_count
    
    # Multiple restarts with different seeds
    best_edges = []
    best_count = 0
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        
        # Greedy fill
        greedy_fill(adj, common, seed)
        
        # Simulated annealing
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        sa_time = remaining * 0.8
        simulated_annealing(adj, common, rng, 5.0, 0.01, sa_time)
        
        # Greedy polish
        greedy_polish(adj, common, rng, 300)
        
        curr_count = get_edge_count(adj)
        if curr_count > best_count:
            best_count = curr_count
            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
    
    # Final verification
    if not best_edges:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            common_count = len(adj_check[u] & adj_check[v])
            if common_count > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return best_edges
