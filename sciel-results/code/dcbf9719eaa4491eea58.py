import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Fallback solution (n=11)
    def get_seed_solution(n):
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    # Deterministic Lazebnik-Woldar construction for n=26
    def get_lazebnik_woldar_26():
        n = 26
        q = 5
        # Map vertex 0 to infinity, 1-25 to (x,y) in F5x2
        # v=0 -> inf
        # v=i (1..25) -> ( (i-1)%5, (i-1)//5 )
        
        def get_coords(v):
            if v == 0:
                return None
            i = v - 1
            return (i % q, i // q)
            
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                ci = get_coords(i)
                cj = get_coords(j)
                
                if ci is None:
                    # i is infinity
                    x, y = cj
                    c = x
                elif cj is None:
                    # j is infinity
                    x, y = ci
                    c = x
                else:
                    x1, y1 = ci
                    x2, y2 = cj
                    if x1 == x2:
                        c = (y1 + y2) % q
                    else:
                        # Slope
                        dx = (x2 - x1) % q
                        dy = (y2 - y1) % q
                        # Inverse of dx in F5
                        # dx is 1..4
                        inv = pow(dx, q - 2, q)
                        c = (dy * inv) % q
                        
                colors.append(c)
                
        return {'n': n, 'colors': colors}

    # Try n=26 first (known to be possible), then 27, 28
    # Use optimized SA with incremental energy updates
    
    # 1. Try deterministic 26
    sol_26 = get_lazebnik_woldar_26()
    if sol_26['n'] == 26:
        # Verify validity just in case (should be fast)
        if _validate(sol_26['n'], sol_26['colors']):
            # Try to improve to 27 with SA
            if time.time() < deadline - 10:
                sol_27 = find_valid_coloring_optimized(27, time_limit=10)
                if sol_27 is not None:
                    return sol_27
            return sol_26
            
    # If deterministic failed (should not), try SA for 26
    for n in [26, 27, 28]:
        if time.time() > deadline - 20:
            break
        solution = find_valid_coloring_optimized(n, time_limit=35)
        if solution is not None:
            return solution
            
    # If we didn't find anything, try smaller n with remaining time
    for n in range(25, 11, -1):
        if time.time() > deadline - 5:
            break
        solution = find_valid_coloring_optimized(n, time_limit=3)
        if solution is not None:
            return solution
            
    return get_seed_solution(11)

def _validate(n, colors):
    """
    Validate that no color class contains a C4.
    """
    if n < 4:
        return True
        
    num_edges = n * (n - 1) // 2
    if len(colors) != num_edges:
        return False
        
    # Build adjacency lists for each color
    adj = [defaultdict(set) for _ in range(5)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    # Check for C4 in each color
    # A C4 exists if there are two vertices u, v with at least 2 common neighbors in color c
    for c in range(5):
        for u in range(n):
            neighbors_u = adj[c][u]
            for v in neighbors_u:
                if v <= u:
                    continue
                # Check common neighbors of u and v in color c
                common = neighbors_u.intersection(adj[c][v])
                if len(common) >= 2:
                    return False
    return True

def find_valid_coloring_optimized(n, time_limit=35):
    """
    Optimized Simulated Annealing for 5-coloring K_n with no C4 in any color.
    Uses incremental energy updates for speed.
    """
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    # Precompute edge indices
    edge_index = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[i][j] = idx
            edge_index[j][i] = idx
            idx += 1
            
    # Initialize random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Bitset adjacency for each color
    # adj_bit[c][u] is integer with bit v set if (u,v) has color c
    adj_bit = [[0] * n for _ in range(5)]
    
    def rebuild_adj_bit():
        for c in range(5):
            for u in range(n):
                adj_bit[c][u] = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index[i][j]]
                adj_bit[c][i] |= (1 << j)
                adj_bit[c][j] |= (1 << i)
                
    rebuild_adj_bit()
    
    # Count common neighbors between u and v in color c
    def common_neighbors(u, v, c):
        return bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
    
    # Energy: sum over all color classes and all pairs of C(2, common_neighbors)
    def compute_energy():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn = common_neighbors(u, v, c)
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
        return total
        
    energy = compute_energy()
    if energy == 0:
        return {'n': n, 'colors': colors}
    
    # SA parameters
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    start_t = time.time()
    
    # Precompute for each edge (i,j) and color c, the list of vertices k
    # that are neighbors of i in color c (excluding j) and neighbors of j in color c (excluding i)
    # This is used for incremental update
    
    def get_neighbors_in_color(u, c, exclude=None):
        """Get list of neighbors of u in color c, optionally excluding a vertex."""
        mask = adj_bit[c][u]
        if exclude is not None:
            mask &= ~(1 << exclude)
        # Convert to list
        neighbors = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            neighbors.append(v)
            m &= m - 1
        return neighbors
    
    # Incremental energy delta when changing edge (i,j) from c1 to c2
    def compute_delta(i, j, c1, c2):
        if c1 == c2:
            return 0
            
        delta = 0
        
        # For color c1: removing edge (i,j)
        # This affects common neighbor counts for pairs (i,k) where k in N_c1(j) \ {i}
        # and pairs (j,k) where k in N_c1(i) \ {j}
        # Also pair (i,j) itself: its common neighbor count doesn't change (edge (i,j) not counted)
        
        Nj_c1 = get_neighbors_in_color(j, c1, exclude=i)
        Ni_c1 = get_neighbors_in_color(i, c1, exclude=j)
        
        for k in Nj_c1:
            old_cn = common_neighbors(i, k, c1)
            new_cn = old_cn - 1
            delta -= (old_cn * (old_cn - 1) // 2) - (new_cn * (new_cn - 1) // 2)
            
        for k in Ni_c1:
            old_cn = common_neighbors(j, k, c1)
            new_cn = old_cn - 1
            delta -= (old_cn * (old_cn - 1) // 2) - (new_cn * (new_cn - 1) // 2)
            
        # For color c2: adding edge (i,j)
        Nj_c2 = get_neighbors_in_color(j, c2, exclude=i)
        Ni_c2 = get_neighbors_in_color(i, c2, exclude=j)
        
        for k in Nj_c2:
            old_cn = common_neighbors(i, k, c2)
            new_cn = old_cn + 1
            delta += (new_cn * (new_cn - 1) // 2) - (old_cn * (old_cn - 1) // 2)
            
        for k in Ni_c2:
            old_cn = common_neighbors(j, k, c2)
            new_cn = old_cn + 1
            delta += (new_cn * (new_cn - 1) // 2) - (old_cn * (old_cn - 1) // 2)
            
        return delta
    
    # Main SA loop
    while time.time() < start_t + time_limit:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        idx_e = edge_index[i][j]
        c1 = colors[idx_e]
        c2 = random.randint(0, 4)
        if c1 == c2:
            continue
            
        dE = compute_delta(i, j, c1, c2)
        
        if dE <= 0 or random.random() < math.exp(-dE / T):
            # Accept move
            colors[idx_e] = c2
            adj_bit[c1][i] &= ~(1 << j)
            adj_bit[c1][j] &= ~(1 << i)
            adj_bit[c2][i] |= (1 << j)
            adj_bit[c2][j] |= (1 << i)
            energy += dE
            
            if energy == 0:
                return {'n': n, 'colors': colors}
                
        T *= cooling
        
    return None
