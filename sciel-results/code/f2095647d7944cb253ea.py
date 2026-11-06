import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Returns a dictionary with 'n' and 'edges'.
    
    Strategy: 
    1. Guaranteed valid graph: 8xK4 (n=32).
    2. Attempt to find a valid graph with n=33 using local search.
    """
    deadline = time.time() + 140.0
    
    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def has_c5(adj, n):
        # Optimized C5 check
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
        
        for u in range(n):
            for v in neighbors[u]:
                if u >= v: continue
                # Path u-a-b-c-v
                for a in neighbors[u]:
                    if a == v: continue
                    for c in neighbors[v]:
                        if c == a or c == u: continue
                        common = neighbors[a].intersection(neighbors[c])
                        if common:
                            for b in common:
                                if b == u or b == v or b == a or b == c:
                                    continue
                                return True
        return False

    def is_valid(adj, n):
        if has_c5(adj, n):
            return False
        # Check alpha <= 8
        def find_ind_set(rem, size):
            if size == 0:
                return True
            if len(rem) < size:
                return False
            u = rem[0]
            new_rem = [v for v in rem[1:] if not adj[u][v]]
            if find_ind_set(new_rem, size - 1):
                return True
            if find_ind_set(rem[1:], size):
                return True
            return False

        if find_ind_set(list(range(n)), 9):
            return False
        return True

    # Base graph: 8xK4
    n_base = 32
    adj = [[0]*35 for _ in range(35)]
    
    # Initialize 8xK4
    for b in range(8):
        start = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[start+i][start+j] = 1
                adj[start+j][start+i] = 1
                
    current_n = 32
    best_adj = [row[:] for row in adj]
    
    # Try to find n=33
    # We know n=32 is valid.
    # We attempt to add vertex 32.
    # To avoid C5, vertex 32 must not form C5.
    # To avoid alpha=9, vertex 32 must be connected to a set S such that alpha(G\S) <= 7.
    
    # Given the difficulty, we just do a random search on the edges of vertex 32
    # and maybe some base edges.
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        adj_try = [row[:] for row in adj]
        # Randomize connections of vertex 32
        for u in range(32):
            if random.random() < 0.5:
                adj_try[32][u] = 1
                adj_try[u][32] = 1
            else:
                adj_try[32][u] = 0
                adj_try[u][32] = 0
        
        # Local search
        T = 1.0
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Flip a random edge
            u = random.randint(0, 32)
            v = random.randint(0, 32)
            if u == v: continue
            
            adj_try[u][v] ^= 1
            adj_try[v][u] ^= 1
            
            if is_valid(adj_try, 33):
                best_adj = [row[:] for row in adj_try]
                current_n = 33
                break
            
            T *= 0.99
        
        if current_n == 33:
            break
            
    return {'n': current_n, 'edges': get_edges_from_adj(best_adj, current_n)}
