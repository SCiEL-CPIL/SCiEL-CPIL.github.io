import random
import time

def construct():
    n = 40
    start_time = time.time()
    time_limit = 115.0
    
    # Bitset representation for fast C6 checking
    # adj[i] is an int where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_c6(u, v):
        """
        Check if adding (u,v) creates a C6.
        A C6 through (u,v) is u-x-y-z-w-v-u.
        x in N(u)\{v}, w in N(v)\{u}.
        y in N(x)\{u,v,x,w}, z in N(w)\{u,v,x,w}.
        Edge (y,z) must exist.
        """
        U = adj[u] & ~(1 << v)
        V = adj[v] & ~(1 << u)
        
        # Iterate x in U
        x = U
        while x:
            lx = x & -x
            x ^= lx
            xi = lx.bit_length() - 1
            
            # Iterate w in V
            # Skip w if w == x (handled by distinctness later, but good to prune)
            w_mask = V & ~(1 << xi) 
            w = w_mask
            while w:
                lw = w & -w
                w ^= lw
                wi = lw.bit_length() - 1
                
                # Candidates for y: neighbors of x, excluding u, v, x, w
                Y_mask = adj[xi] & ~(1 << u) & ~(1 << v) & ~(1 << xi) & ~(1 << wi)
                
                # Candidates for z: neighbors of w, excluding u, v, x, w
                Z_mask = adj[wi] & ~(1 << u) & ~(1 << v) & ~(1 << xi) & ~(1 << wi)
                
                # Check if there is an edge between Y and Z
                # For each y in Y, check if N(y) intersects Z
                # We must ensure y != z
                y = Y_mask
                while y:
                    ly = y & -y
                    y ^= ly
                    yi = ly.bit_length() - 1
                    
                    # Z excluding y
                    Z_eff = Z_mask & ~(1 << yi)
                    if Z_eff & adj[yi]:
                        return True
                        
        return False

    def count_edges():
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edge_list():
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    best_edges = 0
    best_adj = [0] * n
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Strategy: Multiple restarts with Greedy + Local Search (Swaps)
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        # Reset
        adj = [0] * n
        
        # 1. Greedy Phase
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not has_c6(u, v):
                add_edge(u, v)
                
        current_edges = count_edges()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        # 2. Local Search Phase (Swaps to escape local minima)
        # We try to perform swaps that keep edge count constant but change structure.
        # Then we try to add edges again.
        swap_attempts = 0
        max_swaps = 1000
        last_improvement = current_edges
        
        while time.time() - start_time < time_limit and swap_attempts < max_swaps:
            swap_attempts += 1
            
            # Try to find a swap: remove (a,b), add (c,d)
            # We pick a random edge to remove
            u1 = random.randint(0, n-1)
            if adj[u1] == 0:
                continue
            v1 = random.choice([i for i in range(n) if (adj[u1] >> i) & 1 and i != u1])
            
            # Pick a random non-edge to add
            added = False
            for _ in range(20):
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 != v2 and not ((adj[u2] >> v2) & 1):
                    # Ensure we don't swap an edge with itself
                    if (u1, v1) != (u2, v2) and (u1, v1) != (v2, u2):
                        break
            else:
                continue
                
            # Try the swap
            remove_edge(u1, v1)
            if not has_c6(u2, v2):
                add_edge(u2, v2)
                # Swap accepted (edge count same)
                # Check if we can now add more edges
                improved = False
                # Try to add random edges
                for _ in range(50):
                    u_add = random.randint(0, n-1)
                    v_add = random.randint(0, n-1)
                    if u_add != v_add and not ((adj[u_add] >> v_add) & 1):
                        if not has_c6(u_add, v_add):
                            add_edge(u_add, v_add)
                            improved = True
                            
                if improved:
                    current_edges = count_edges()
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                    if current_edges > last_improvement:
                        last_improvement = current_edges
                        swap_attempts = 0 # Reset counter if we improved
            else:
                # Swap failed, restore
                add_edge(u1, v1)
                
    return get_edge_list()
