import time
import random

def construct():
    """
    Finds a 4-coloring of K_n with no monochromatic triangles, maximizing n.
    Combines:
    1. Efficient WalkSAT local search with delta updates (from current program).
    2. Vertex expansion with backtracking (from inspiration 2) to grow n.
    3. Strategic time allocation and randomization.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit

    # Fallback: The Greenwood-Gleason coloring for n=32
    def get_seed_solution():
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    seed_sol = get_seed_solution()
    best_n = 32
    best_colors = seed_sol['colors']

    # Local Search Solver for a given N
    # Returns the flat list of colors if a valid coloring is found, else None
    def try_solve(N, time_limit):
        if N < 4:
            return None
        
        # Initialize random coloring
        c = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                c[i][j] = random.randint(0, 3)
        
        # Precompute A[c][u][v] = count of w such that c[u][w] == c and c[v][w] == c
        A = [[[0] * N for _ in range(N)] for _ in range(4)]
        
        # Initial calculation of A
        for col in range(4):
            M = [[0] * N for _ in range(N)]
            for u in range(N):
                for w in range(N):
                    if u == w: continue
                    if u < w:
                        if c[u][w] == col: M[u][w] = 1
                    else:
                        if c[w][u] == col: M[u][w] = 1
            
            for u in range(N):
                for v in range(u + 1, N):
                    count = 0
                    for w in range(N):
                        if w == u or w == v: continue
                        if M[u][w] and M[v][w]:
                            count += 1
                    A[col][u][v] = count
                    A[col][v][u] = count
        
        total_bad = sum(A[col][u][v] for col in range(4) for u in range(N) for v in range(u + 1, N))
        
        if total_bad == 0:
            flat = []
            for i in range(N):
                for j in range(i + 1, N):
                    flat.append(c[i][j])
            return flat

        start_time = time.time()
        # Parameters for WalkSAT
        p_flip = 0.5 

        while time.time() < start_time + time_limit:
            # 1. Find best move (minimizes total_bad)
            best_delta = 1000000
            best_u, best_v, best_col = -1, -1, -1
            
            for u in range(N):
                for v in range(u + 1, N):
                    old_col = c[u][v]
                    for new_col in range(4):
                        if new_col == old_col: continue
                        delta = A[new_col][u][v] - A[old_col][u][v]
                        if delta < best_delta:
                            best_delta = delta
                            best_u, best_v, best_col = u, v, new_col
            
            if best_u == -1:
                # Random move
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v: continue
                if u > v: u, v = v, u
                old_col = c[u][v]
                new_col = random.choice([0, 1, 2, 3])
                if new_col == old_col: continue
                best_delta = A[new_col][u][v] - A[old_col][u][v]
                best_col = new_col
                best_u, best_v = u, v
            else:
                if best_delta > 0 and random.random() < p_flip:
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    old_col = c[u][v]
                    new_col = random.choice([0, 1, 2, 3])
                    if new_col == old_col: continue
                    best_delta = A[new_col][u][v] - A[old_col][u][v]
                    best_col = new_col
                    best_u, best_v = u, v

            # Apply Move
            u, v = best_u, best_v
            old_col = c[u][v]
            new_col = best_col
            c[u][v] = new_col
            
            # Update A
            for w in range(N):
                if w == u or w == v: continue
                
                col_vw = c[v][w] if v < w else c[w][v]
                if col_vw == old_col:
                    A[old_col][u][w] -= 1
                    A[old_col][w][u] -= 1
                if col_vw == new_col:
                    A[new_col][u][w] += 1
                    A[new_col][w][u] += 1
                    
                col_uw = c[u][w] if u < w else c[w][u]
                if col_uw == old_col:
                    A[old_col][v][w] -= 1
                    A[old_col][w][v] -= 1
                if col_uw == new_col:
                    A[new_col][v][w] += 1
                    A[new_col][w][v] += 1
            
            total_bad += best_delta
            
            if total_bad == 0:
                flat = []
                for i in range(N):
                    for j in range(i + 1, N):
                        flat.append(c[i][j])
                return flat

        return None

    # Strategy:
    # 1. Try to find a valid coloring for n=50 (the known record).
    # 2. If successful, try to extend to n=51 using backtracking.
    # 3. If extension fails, return n=50.
    # 4. If n=50 search fails, try n=49, etc.
    # 5. If all fail, return seed (n=32).

    # Helper to add a vertex to a valid coloring
    def try_add_vertex(n, c_flat, c_2d):
        """
        c_flat: list of colors for K_n
        c_2d: 2D list of colors for K_n
        Returns (new_flat, new_2d) if successful, else (None, None)
        """
        # Backtracking for the new vertex (index n)
        # We need to assign colors to (i, n) for i in 0..n-1
        # Constraint: for any i < j < n, c[i][j] != c[i][n] != c[j][n]
        
        # Precompute constraints
        # For each i, the forbidden colors for c[i][n] depend on c[j][n] for j < i?
        # No, it's a global constraint.
        # c[i][n] cannot be c[i][j] if c[j][n] == c[i][j].
        
        # Let's just do simple backtracking.
        # Order: 0 to n-1
        assignments = [0] * n
        
        def backtrack(i):
            if i == n:
                return True
            
            # Try colors 0,1,2,3
            # Heuristic: Try colors that are least conflicting?
            # For now, random order or fixed? Fixed 0,1,2,3 is fine.
            for col in range(4):
                assignments[i] = col
                # Check conflicts with previous assignments
                valid = True
                for j in range(i):
                    # Triangle (j, i, n)
                    # c[j][i], c[i][n] (which is col), c[j][n] (which is assignments[j])
                    if c_2d[j][i] == col == assignments[j]:
                        valid = False
                        break
                if valid:
                    if backtrack(i + 1):
                        return True
            return False
        
        if backtrack(0):
            # Success
            new_2d = [[0] * (n + 1) for _ in range(n + 1)]
            for i in range(n):
                for j in range(n):
                    new_2d[i][j] = c_2d[i][j]
                new_2d[i][n] = assignments[i]
                new_2d[n][i] = assignments[i]
            
            new_flat = c_flat[:]
            for i in range(n):
                new_flat.append(assignments[i])
            
            return new_flat, new_2d
        return None, None

    # Main search loop
    # Try 50 first
    if time.time() < deadline - 10.0:
        t_limit = 30.0
        sol = try_solve(50, t_limit)
        if sol is not None:
            # Build 2D array
            c_2d = [[0] * 50 for _ in range(50)]
            idx = 0
            for i in range(50):
                for j in range(i + 1, 50):
                    c_2d[i][j] = sol[idx]
                    c_2d[j][i] = sol[idx]
                    idx += 1
            
            # Try to add vertex 50 (making n=51)
            if time.time() < deadline - 5.0:
                new_flat, new_2d = try_add_vertex(50, sol, c_2d)
                if new_flat is not None:
                    return {'n': 51, 'colors': new_flat}
            
            # If n=51 failed, return n=50
            return {'n': 50, 'colors': sol}
    
    # If 50 failed, try 49
    if time.time() < deadline - 10.0:
        t_limit = 20.0
        sol = try_solve(49, t_limit)
        if sol is not None:
            c_2d = [[0] * 49 for _ in range(49)]
            idx = 0
            for i in range(49):
                for j in range(i + 1, 49):
                    c_2d[i][j] = sol[idx]
                    c_2d[j][i] = sol[idx]
                    idx += 1
            
            if time.time() < deadline - 5.0:
                new_flat, new_2d = try_add_vertex(49, sol, c_2d)
                if new_flat is not None:
                    return {'n': 50, 'colors': new_flat}
            
            return {'n': 49, 'colors': sol}

    # Fallback to seed
    return seed_sol
