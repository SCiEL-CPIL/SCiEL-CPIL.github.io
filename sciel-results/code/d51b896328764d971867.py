import random
import time

def construct():
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16) (the Clebsch graph three times),
    # doubled: two copies with colour 3 on all edges between them (a bipartite colour class).
    # This is the seed solution for n=32. We will try to exceed this.
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_n = 32
    best_colors = None
    
    # Initialize seed solution for n=32 to have a fallback
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    best_colors = seed_colors
    
    # Build the seed adjacency matrix for n=32
    def build_seed_matrix(n=32):
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = seed_colors[idx]
                mat[i][j] = c
                mat[j][i] = c
                idx += 1
        return mat
    
    # Compute slack for each edge: slack(i,j) = number of k != i,j such that 
    # NOT (mat[i][k] == mat[i][j] AND mat[j][k] == mat[i][j])
    # Equivalently, for edge (i,j) of color c, slack = (n-2) - count of k where mat[i][k]==c and mat[j][k]==c
    def compute_slacks(mat, n):
        slacks = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                common = 0
                for k in range(n):
                    if k != i and k != j:
                        if mat[i][k] == c and mat[j][k] == c:
                            common += 1
                slacks[i][j] = (n - 2) - common
                slacks[j][i] = slacks[i][j]
        return slacks
    
    # Greedy extension: add vertices one by one, choosing colors to minimize triangles
    # and maximize minimum slack
    def greedy_extend(mat, n, target_n, slacks):
        if n >= target_n:
            return n, mat, slacks
        
        while n < target_n:
            if time.time() - start_time > time_limit:
                return n, mat, slacks
            
            # Try to add vertex n
            # For each possible color assignment to edges (n, 0), ..., (n, n-1),
            # count monochromatic triangles involving n
            # We want an assignment with 0 triangles that maximizes min slack
            
            # Greedy: for each edge (n, u), pick the color that minimizes triangles
            # Use local search to find a good assignment
            
            new_colors = [random.randint(0, 3) for _ in range(n)]
            best_tris = float('inf')
            best_assignment = None
            
            # Local search
            for iteration in range(200):
                # Count triangles involving n
                tris = 0
                for u in range(n):
                    cu = new_colors[u]
                    for w in range(u + 1, n):
                        if cu == new_colors[w] == mat[u][w]:
                            tris += 1
                
                if tris < best_tris:
                    best_tris = tris
                    best_assignment = new_colors[:]
                
                if tris == 0:
                    break
                
                # Pick a random vertex u and change its color
                u = random.randint(0, n - 1)
                new_colors[u] = random.choice([c for c in range(4) if c != new_colors[u]])
            
            if best_tris == 0:
                # Commit the assignment
                for u in range(n):
                    mat[n][u] = best_assignment[u]
                    mat[u][n] = best_assignment[u]
                n += 1
                # Recompute slacks
                slacks = compute_slacks(mat, n)
            else:
                # Stuck
                return n, mat, slacks
        
        return n, mat, slacks
    
    # Local search on the entire coloring to try to add more vertices
    def local_search_extend(mat, n, target_n, slacks):
        if n >= target_n:
            return n, mat, slacks
        
        # Try to find a vertex that can be added
        # For each candidate vertex position, try all 4^n colorings? No, too many.
        # Instead, try adding one vertex at a time with local search
        
        while n < target_n:
            if time.time() - start_time > time_limit:
                return n, mat, slacks
            
            # Try to add vertex n
            new_colors = [random.randint(0, 3) for _ in range(n)]
            best_tris = float('inf')
            best_assignment = None
            
            for iteration in range(500):
                tris = 0
                for u in range(n):
                    cu = new_colors[u]
                    for w in range(u + 1, n):
                        if cu == new_colors[w] == mat[u][w]:
                            tris += 1
                
                if tris < best_tris:
                    best_tris = tris
                    best_assignment = new_colors[:]
                
                if tris == 0:
                    break
                
                # Simulated annealing-like: accept worse solutions sometimes
                u = random.randint(0, n - 1)
                old_c = new_colors[u]
                new_c = random.choice([c for c in range(4) if c != old_c])
                new_colors[u] = new_c
            
            if best_tris == 0:
                for u in range(n):
                    mat[n][u] = best_assignment[u]
                    mat[u][n] = best_assignment[u]
                n += 1
                slacks = compute_slacks(mat, n)
            else:
                # Try perturbing existing edges and retry
                # Pick a random edge and change its color
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i == j:
                    continue
                old_c = mat[i][j]
                new_c = random.choice([c for c in range(4) if c != old_c])
                mat[i][j] = new_c
                mat[j][i] = new_c
                # Recompute slacks
                slacks = compute_slacks(mat, n)
                # Check if we still have no triangles
                has_tri = False
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            if mat[a][b] == mat[a][c] == mat[b][c]:
                                has_tri = True
                                break
                        if has_tri:
                            break
                    if has_tri:
                        break
                if has_tri:
                    # Revert
                    mat[i][j] = old_c
                    mat[j][i] = old_c
                    slacks = compute_slacks(mat, n)
                    return n, mat, slacks
        
        return n, mat, slacks
    
    # Main loop: try multiple strategies
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(attempt * 12345 + 678)
        
        # Start from seed K32
        mat = build_seed_matrix(32)
        slacks = compute_slacks(mat, 32)
        
        # Try greedy first
        n, mat, slacks = greedy_extend(mat, 32, 50, slacks)
        
        # If stuck, try local search
        if n < 50 and time.time() - start_time < time_limit:
            n, mat, slacks = local_search_extend(mat, n, 50, slacks)
        
        # Update best
        if n > best_n:
            res_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    res_colors.append(mat[i][j])
            best_n = n
            best_colors = res_colors
        
        if best_n >= 50:
            break
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
