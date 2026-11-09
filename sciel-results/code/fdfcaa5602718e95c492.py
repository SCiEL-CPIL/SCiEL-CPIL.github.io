import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # safety margin before 120s kill
    
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
    
    def exhaustive_polish(adj, common):
        """Try to add every possible missing edge. Repeat until no more can be added."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u] and can_add(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        improved = True
    
    def greedy_fill(adj, common, seed):
        rng = random.Random(seed)
        # Try all pairs in random order
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if can_add(adj, common, u, v):
                add_edge(adj, common, u, v)
        # Multiple passes to catch more
        for _ in range(5):
            rng.shuffle(pairs)
            for u, v in pairs:
                if v not in adj[u] and can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
    
    def simulated_annealing(adj, common, rng, time_budget):
        current_count = get_edge_count(adj)
        sa_end_time = start_time + time_budget
        T = 2.0
        cooling = 0.999
        
        while (time.time() - start_time) < sa_end_time:
            if T < 0.01:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a random missing edge
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or v in adj[u]:
                    continue
                if can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_count += 1
                    
            elif move_type < 0.7:
                # Remove a random edge (accept with probability)
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if not edges_list:
                    continue
                u, v = rng.choice(edges_list)
                remove_edge(adj, common, u, v)
                delta = -1
                if rng.random() < math.exp(delta / T):
                    current_count -= 1
                else:
                    add_edge(adj, common, u, v)
                    
            else:
                # Swap: remove one edge, try to add another
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if not edges_list:
                    continue
                u1, v1 = rng.choice(edges_list)
                
                # Try a few candidates for the new edge
                found_swap = False
                for _ in range(20):
                    u2 = rng.randint(0, n - 1)
                    v2 = rng.randint(0, n - 1)
                    if u2 == v2 or v2 in adj[u2]:
                        continue
                    if {u2, v2} == {u1, v1}:
                        continue
                    # Check if we can do the swap
                    remove_edge(adj, common, u1, v1)
                    if can_add(adj, common, u2, v2):
                        add_edge(adj, common, u2, v2)
                        found_swap = True
                        break
                    else:
                        add_edge(adj, common, u1, v1)
                
                if not found_swap:
                    # Try accepting a non-improving swap with probability
                    for _ in range(5):
                        u2 = rng.randint(0, n - 1)
                        v2 = rng.randint(0, n - 1)
                        if u2 == v2 or v2 in adj[u2]:
                            continue
                        if {u2, v2} == {u1, v1}:
                            continue
                        remove_edge(adj, common, u1, v1)
                        if can_add(adj, common, u2, v2):
                            add_edge(adj, common, u2, v2)
                            # Accept this swap (it's neutral in count)
                            break
                        else:
                            add_edge(adj, common, u1, v1)
                            if rng.random() < math.exp(-1.0 / T):
                                # Force accept a bad swap to escape local optima
                                # This won't actually work since we can't add invalid edges
                                pass
                            break

            T *= cooling
        
        return current_count
    
    def structured_init(adj, common):
        """Initialize with a structured pattern that's likely K2,3-free."""
        # Use a circulant-like structure: connect i to i+1, i+2, i+3, i+4 mod 20
        # But this might create K2,3, so let's use a sparser base
        # Connect each vertex to its next 3 neighbors in a cycle
        for i in range(n):
            for d in [1, 2, 3]:
                j = (i + d) % n
                if i < j and can_add(adj, common, i, j):
                    add_edge(adj, common, i, j)
        # Add some cross edges
        for i in range(n):
            for d in [5, 7]:
                j = (i + d) % n
                if i < j and can_add(adj, common, i, j):
                    add_edge(adj, common, i, j)
    
    # Multiple restarts with different seeds
    best_edges = []
    best_count = 0
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555, 666, 777]
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        
        # Initialize: either structured or greedy
        if seed % 2 == 0:
            structured_init(adj, common)
        else:
            greedy_fill(adj, common, seed)
        
        # Simulated annealing
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            break
        sa_time = remaining * 0.7
        simulated_annealing(adj, common, rng, sa_time)
        
        # Exhaustive polish: try to add every possible edge
        exhaustive_polish(adj, common)
        
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
