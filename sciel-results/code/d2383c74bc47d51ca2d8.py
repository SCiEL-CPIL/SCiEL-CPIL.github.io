import time
import random

def construct():
    """
    Finds a 4-coloring of K_n with no monochromatic triangles, maximizing n.
    Uses a local search (WalkSAT-style) with efficient delta updates.
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
    def try_solve(N, time_limit):
        if N < 4:
            return None
        
        # Initialize random coloring
        # colors[i][j] for i < j
        # We use a flat list or 2D array. 2D is easier for logic.
        # c[u][v] where u < v.
        c = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                c[i][j] = random.randint(0, 3)
        
        # Precompute A[c][u][v] = count of w such that c[u][w] == c and c[v][w] == c
        # A is a list of 4 2D arrays.
        # A[col][u][v]
        A = [[[0] * N for _ in range(N)] for _ in range(4)]
        
        # Initial calculation of A
        # For each color col, for each pair (u, v), count w.
        # This is O(N^3 * 4) which is fine for N=50 (50^3 * 4 = 500,000)
        for col in range(4):
            # Create a matrix M where M[u][w] = 1 if c[u][w] == col else 0
            # A[col][u][v] is sum of M[u][w] * M[v][w] for w != u, v
            # Actually, since c[u][u] is undefined, we just iterate w.
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
                    A[col][v][u] = count # Symmetry for easier access
        
        total_bad = sum(A[col][u][v] for col in range(4) for u in range(N) for v in range(u + 1, N))
        
        if total_bad == 0:
            # Convert to flat list
            flat = []
            for i in range(N):
                for j in range(i + 1, N):
                    flat.append(c[i][j])
            return flat

        start_time = time.time()
        steps = 0
        # Parameters for WalkSAT
        p_flip = 0.5 # Probability to pick a random variable if no improving move
        
        while time.time() < start_time + time_limit:
            # 1. Find best move (minimizes total_bad)
            best_delta = 1000000
            best_u, best_v, best_col = -1, -1, -1
            
            # Sample a subset of edges if N is large? For N=50, N^2/2 = 1250 edges.
            # Checking 4 colors for each edge is 5000 checks.
            # 5000 * O(1) is very fast. We can check all edges.
            for u in range(N):
                for v in range(u + 1, N):
                    old_col = c[u][v]
                    for new_col in range(4):
                        if new_col == old_col: continue
                        # Delta = A[new_col][u][v] - A[old_col][u][v]
                        delta = A[new_col][u][v] - A[old_col][u][v]
                        if delta < best_delta:
                            best_delta = delta
                            best_u, best_v, best_col = u, v, new_col
            
            if best_u == -1:
                # No moves? (Should not happen if N>=4 and colors<4, but just in case)
                # Or all deltas are positive?
                # If best_delta > 0, we are stuck in a local minimum.
                # Pick a random move.
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v: continue
                if u > v: u, v = v, u
                old_col = c[u][v]
                new_col = random.choice([0, 1, 2, 3])
                if new_col == old_col: continue
                best_delta = A[new_col][u][v] - A[old_col][u][v]
                best_col = new_col
            else:
                # If best_delta is still very high or positive, might be stuck.
                # We accept the best move even if it increases bad count (Simulated Annealing style or just greedy).
                # Standard WalkSAT picks the best if it improves, otherwise random.
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
                else:
                    # Use the best move found
                    pass # best_u, best_v, best_col are already set

            # Apply Move
            old_col = c[best_u][best_v]
            new_col = best_col
            
            # Update c
            c[best_u][best_v] = new_col
            
            # Update A
            # For all w != u, v:
            # A[col][u][w] changes if c[v][w] == col
            # A[col][v][w] changes if c[u][w] == col
            
            u, v = best_u, best_v
            
            for w in range(N):
                if w == u or w == v: continue
                
                # Update A for edge (u, w)
                # The witness is v.
                # c[v][w] is fixed.
                col_vw = c[v][w] if v < w else c[w][v]
                
                # If old_col was the color of (u, v), it contributed to A[old_col][u][w] if c[v][w]==old_col?
                # No, A[col][u][w] counts z such that c[u][z]==col and c[w][z]==col.
                # z=v is a witness if c[u][v]==col and c[w][v]==col.
                # We changed c[u][v] from old_col to new_col.
                
                if col_vw == old_col:
                    A[old_col][u][w] -= 1
                    A[old_col][w][u] -= 1
                if col_vw == new_col:
                    A[new_col][u][w] += 1
                    A[new_col][w][u] += 1
                    
                # Update A for edge (v, w)
                # Witness is u.
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
            
            steps += 1

        return None

    # Try to find solutions for N > 32
    # Start from 50 down to 33
    # Allocate time budget
    # Total time ~110s.
    # Try 50 with 30s, 49 with 20s, 48 with 15s, etc.
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    time_per_target = 5.0 # Base time
    
    for N in targets:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate more time to larger N initially?
        # Or just fixed small chunks.
        # If we find a solution, we return immediately.
        # Since 50 is known to exist, it should be found quickly.
        # If 50 is hard, we might waste time.
        # Let's give 10s to 50, 5s to others.
        t_limit = 10.0 if N == 50 else 5.0
        
        if time.time() + t_limit > deadline:
            t_limit = deadline - time.time() - 1.0
            if t_limit < 0: break
            
        sol = try_solve(N, t_limit)
        if sol is not None:
            return {'n': N, 'colors': sol}

    # If no better solution found, return seed
    return seed_sol
