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
            # Try to improve to 27 with backtracking first (more targeted), then SA
            if time.time() < deadline - 15:
                sol_27_bt = backtracking_search(27, time_limit=20)
                if sol_27_bt is not None:
                    return sol_27_bt
            if time.time() < deadline - 10:
                sol_27 = find_valid_coloring_optimized(27, time_limit=15)
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

def backtracking_search(n, time_limit=20):
    """
    Backtracking search with forward-checking for 5-coloring K_n with no C4.
    Assigns edges in a structured order, pruning when a C4 is about to form.
    """
    start_t = time.time()
    num_edges = n * (n - 1) // 2
    
    # Edge indexing
    edge_index = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[i][j] = idx
            edge_index[j][i] = idx
            idx += 1
    
    # Process edges in order: for each vertex i, assign edges (i, j) for j > i
    # This creates a natural ordering
    
    # State: colors array, and for each color c, adjacency bitmasks
    colors = [-1] * num_edges
    adj_bit = [[0] * n for _ in range(5)]
    
    # For forward checking: when we assign edge (i,j) color c, we need to check
    # that no C4 is formed. A C4 forms if there exist u,v such that both u and v
    # are neighbors of i and j in color c (excluding the direct edge).
    
    # Precompute: for each pair (i,j), the list of colors that would create a C4
    # if assigned to edge (i,j), given current state
    
    def check_c4_conflict(i, j, c):
        """Check if assigning color c to edge (i,j) creates a C4."""
        # After adding (i,j) in color c, check if any pair has 2+ common neighbors in color c
        # The new edge affects common neighbors of:
        # - (i, k) for k in N_c(j) \ {i}: i gains neighbor j, so common(i,k) might increase
        # - (j, k) for k in N_c(i) \ {j}: j gains neighbor i, so common(j,k) might increase
        # - (i, j) itself: common(i,j) in color c = |N_c(i) ∩ N_c(j)| \ {i,j}
        
        # Check (i,j): common neighbors in color c
        common_ij = adj_bit[c][i] & adj_bit[c][j]
        # Remove i and j from consideration (they're not neighbors of themselves)
        common_ij &= ~(1 << i)
        common_ij &= ~(1 << j)
        if bin(common_ij).count('1') >= 2:
            return True
        
        # Check (i, k) for k in N_c(j) \ {i}
        mask_j = adj_bit[c][j] & ~(1 << i)
        m = mask_j
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            if k == i:
                continue
            # common(i,k) in color c would increase by 1 (j is now neighbor of i in color c)
            common_ik = (adj_bit[c][i] & adj_bit[c][k]) & ~(1 << i) & ~(1 << k)
            if bin(common_ik).count('1') >= 2:
                return True
        
        # Check (j, k) for k in N_c(i) \ {j}
        mask_i = adj_bit[c][i] & ~(1 << j)
        m = mask_i
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            if k == j:
                continue
            # common(j,k) in color c would increase by 1 (i is now neighbor of j in color c)
            common_jk = (adj_bit[c][j] & adj_bit[c][k]) & ~(1 << j) & ~(1 << k)
            if bin(common_jk).count('1') >= 2:
                return True
        
        return False
    
    # Backtracking with randomized restarts
    best_energy = float('inf')
    
    for restart in range(50):
        if time.time() > start_t + time_limit:
            break
            
        # Randomized initialization for some edges to break symmetry
        # Use a partial random coloring as starting point
        for e in range(num_edges):
            colors[e] = random.randint(0, 4)
        
        # Rebuild adj_bit
        for c in range(5):
            for u in range(n):
                adj_bit[c][u] = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index[i][j]]
                adj_bit[c][i] |= (1 << j)
                adj_bit[c][j] |= (1 << i)
        
        # Now do local repair: iteratively fix C4 violations
        # This is a hybrid approach: start random, then repair
        repaired = local_repair(n, colors, adj_bit, edge_index, time_limit=3)
        if repaired is not None:
            return {'n': n, 'colors': repaired}
    
    return None

def local_repair(n, colors, adj_bit, edge_index, time_limit=3):
    """
    Local repair: iteratively fix C4 violations by changing edge colors.
    """
    start_t = time.time()
    num_edges = n * (n - 1) // 2
    
    def count_c4s():
        """Count total C4 violations."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
        return total
    
    violations = count_c4s()
    if violations == 0:
        return colors[:]
    
    T = 2.0
    cooling = 0.999
    
    while time.time() < start_t + time_limit:
        if T < 0.01:
            T = 0.01
            
        # Find a C4 violation and try to fix it
        # Pick a random edge that's part of a violation
        found = False
        for _ in range(100):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            e_idx = edge_index[i][j]
            c1 = colors[e_idx]
            
            # Check if this edge is in a C4
            in_c4 = False
            for c in range(5):
                if c != c1:
                    continue
                # Check if changing this edge would help
                # For now, just try random recoloring
                in_c4 = True
                break
            
            if not in_c4:
                continue
            
            c2 = random.randint(0, 4)
            if c1 == c2:
                continue
            
            # Compute delta
            delta = compute_delta_fast(i, j, c1, c2, adj_bit, n)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Apply
                colors[e_idx] = c2
                adj_bit[c1][i] &= ~(1 << j)
                adj_bit[c1][j] &= ~(1 << i)
                adj_bit[c2][i] |= (1 << j)
                adj_bit[c2][j] |= (1 << i)
                found = True
                break
        
        if not found:
            T *= cooling
        
        violations = count_c4s()
        if violations == 0:
            return colors[:]
        
        T *= cooling
    
    return None

def compute_delta_fast(i, j, c1, c2, adj_bit, n):
    """Fast delta computation for energy change."""
    if c1 == c2:
        return 0
    
    delta = 0
    
    # Removing (i,j) from color c1
    mask_j = adj_bit[c1][j] & ~(1 << i)
    m = mask_j
    while m:
        lsb = m & (-m)
        k = lsb.bit_length() - 1
        m &= m - 1
        old_cn = bin(adj_bit[c1][i] & adj_bit[c1][k]).count('1')
        new_cn = old_cn - 1
        if old_cn >= 2:
            delta -= old_cn * (old_cn - 1) // 2
        if new_cn >= 2:
            delta += new_cn * (new_cn - 1) // 2
    
    mask_i = adj_bit[c1][i] & ~(1 << j)
    m = mask_i
    while m:
        lsb = m & (-m)
        k = lsb.bit_length() - 1
        m &= m - 1
        old_cn = bin(adj_bit[c1][j] & adj_bit[c1][k]).count('1')
        new_cn = old_cn - 1
        if old_cn >= 2:
            delta -= old_cn * (old_cn - 1) // 2
        if new_cn >= 2:
            delta += new_cn * (new_cn - 1) // 2
    
    # Adding (i,j) to color c2
    mask_j = adj_bit[c2][j] & ~(1 << i)
    m = mask_j
    while m:
        lsb = m & (-m)
        k = lsb.bit_length() - 1
        m &= m - 1
        old_cn = bin(adj_bit[c2][i] & adj_bit[c2][k]).count('1')
        new_cn = old_cn + 1
        if old_cn >= 2:
            delta -= old_cn * (old_cn - 1) // 2
        if new_cn >= 2:
            delta += new_cn * (new_cn - 1) // 2
    
    mask_i = adj_bit[c2][i] & ~(1 << j)
    m = mask_i
    while m:
        lsb = m & (-m)
        k = lsb.bit_length() - 1
        m &= m - 1
        old_cn = bin(adj_bit[c2][j] & adj_bit[c2][k]).count('1')
        new_cn = old_cn + 1
        if old_cn >= 2:
            delta -= old_cn * (old_cn - 1) // 2
        if new_cn >= 2:
            delta += new_cn * (new_cn - 1) // 2
    
    return delta

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
    Uses incremental energy updates for speed. Includes final polish pass.
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
    
    # Energy: sum over all color classes and all pairs of C(2, common_neighbors)
    def compute_energy():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
        return total
        
    energy = compute_energy()
    if energy == 0:
        return {'n': n, 'colors': colors}
    
    # SA parameters - tuned for faster convergence
    T = 3.0
    cooling = 0.9997
    min_T = 0.05
    start_t = time.time()
    polish_time = min(5, time_limit * 0.2)  # Reserve time for polish
    
    # Incremental energy delta when changing edge (i,j) from c1 to c2
    def compute_delta(i, j, c1, c2):
        if c1 == c2:
            return 0
            
        delta = 0
        
        # For color c1: removing edge (i,j)
        mask_j = adj_bit[c1][j] & ~(1 << i)
        m = mask_j
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            old_cn = bin(adj_bit[c1][i] & adj_bit[c1][k]).count('1')
            new_cn = old_cn - 1
            if old_cn >= 2:
                delta -= old_cn * (old_cn - 1) // 2
            if new_cn >= 2:
                delta += new_cn * (new_cn - 1) // 2
        
        mask_i = adj_bit[c1][i] & ~(1 << j)
        m = mask_i
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            old_cn = bin(adj_bit[c1][j] & adj_bit[c1][k]).count('1')
            new_cn = old_cn - 1
            if old_cn >= 2:
                delta -= old_cn * (old_cn - 1) // 2
            if new_cn >= 2:
                delta += new_cn * (new_cn - 1) // 2
        
        # For color c2: adding edge (i,j)
        mask_j = adj_bit[c2][j] & ~(1 << i)
        m = mask_j
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            old_cn = bin(adj_bit[c2][i] & adj_bit[c2][k]).count('1')
            new_cn = old_cn + 1
            if old_cn >= 2:
                delta -= old_cn * (old_cn - 1) // 2
            if new_cn >= 2:
                delta += new_cn * (new_cn - 1) // 2
        
        mask_i = adj_bit[c2][i] & ~(1 << j)
        m = mask_i
        while m:
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            m &= m - 1
            old_cn = bin(adj_bit[c2][j] & adj_bit[c2][k]).count('1')
            new_cn = old_cn + 1
            if old_cn >= 2:
                delta -= old_cn * (old_cn - 1) // 2
            if new_cn >= 2:
                delta += new_cn * (new_cn - 1) // 2
        
        return delta
    
    # Main SA loop
    sa_deadline = start_t + time_limit - polish_time
    while time.time() < sa_deadline:
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
        
    # Final polish pass: aggressive local search at low temperature
    if time.time() < start_t + time_limit:
        polish_deadline = start_t + time_limit
        T_polish = 0.5
        cooling_polish = 0.999
        
        while time.time() < polish_deadline:
            if T_polish < 0.01:
                T_polish = 0.01
            
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
            
            if dE <= 0 or random.random() < math.exp(-dE / T_polish):
                colors[idx_e] = c2
                adj_bit[c1][i] &= ~(1 << j)
                adj_bit[c1][j] &= ~(1 << i)
                adj_bit[c2][i] |= (1 << j)
                adj_bit[c2][j] |= (1 << i)
                energy += dE
                
                if energy == 0:
                    return {'n': n, 'colors': colors}
                    
            T_polish *= cooling_polish
    
    return None
