import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Try to find a valid graph on n vertices
    for n in [39, 38, 37, 36]:
        result = try_construct(n, deadline)
        if result:
            return result

    # Fallback to a known valid smaller graph
    # Ten disjoint triangles: n=30, alpha=10
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def try_construct(n, deadline):
    neighbors = [0] * n
    
    def can_add_edge(u, v):
        if (neighbors[u] >> v) & 1:
            return False
        if neighbors[u] & neighbors[v]:
            return False
        return True

    def add_edge(u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)

    def has_independent_set_of_size(k):
        # k is the target size, e.g., 11
        # Returns True if found, False if proven not to exist, None if timeout
        
        def bnb(vertices, current_size):
            nonlocal deadline
            if time.time() > deadline:
                raise TimeoutError()
            
            if current_size >= k:
                return True
            
            if not vertices:
                return False
            
            if current_size + vertices.bit_count() < k:
                return False
            
            # Pick vertex with highest degree in induced subgraph
            max_deg = -1
            pick = -1
            temp = vertices
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (neighbors[v] & vertices).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pick = v
            
            # Branch 1: Include pick
            new_vertices = vertices & ~(neighbors[pick] | (1 << pick))
            if bnb(new_vertices, current_size + 1):
                return True
            
            # Branch 2: Exclude pick
            new_vertices = vertices & ~(1 << pick)
            if bnb(new_vertices, current_size):
                return True
                
            return False

        try:
            return bnb((1 << n) - 1, 0)
        except TimeoutError:
            return None

    def find_greedy_independent_set(vertices):
        remaining = vertices
        s = set()
        while remaining:
            min_deg = float('inf')
            pick = -1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (neighbors[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    pick = v
            s.add(pick)
            remaining &= ~(neighbors[pick] | (1 << pick))
        return s

    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        neighbors = [0] * n
        
        # 1. Greedy Random Edge Addition
        potential_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if can_add_edge(u, v):
                add_edge(u, v)
            if time.time() > deadline:
                break
        
        # 2. Local Search
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            has_11 = has_independent_set_of_size(11)
            
            if has_11 is False:
                edges = []
                for i in range(n):
                    mask = neighbors[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}
            
            if has_11 is None:
                break # Timeout in checker, assume bad

            S = find_greedy_independent_set((1 << n) - 1)
            
            if len(S) < 11:
                # Greedy says < 11, but B&B says 11 exists?
                # This shouldn't happen if B&B is correct.
                # If B&B is correct and finds 11, Greedy might have missed it or found a smaller one.
                # We trust B&B.
                pass

            added = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u = S_list[i]
                    v = S_list[j]
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        added = True
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
                    
            if not added:
                improved = False
            else:
                improved = True
                
    return None
