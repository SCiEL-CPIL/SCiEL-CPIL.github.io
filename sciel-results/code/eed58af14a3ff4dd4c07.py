import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate the initial N=32 coloring from the seed
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
    
    current_colors = []
    N = 32
    for i in range(N):
        for j in range(i + 1, N):
            current_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    def get_idx(i, j, M):
        # Index for pair (i, j) in K_M, i < j
        return i * (M - 1) - i * (i + 1) // 2 + (j - i - 1)

    def is_valid_coloring(colors, N):
        # Check for monochromatic triangles
        for i in range(N):
            for j in range(i + 1, N):
                c_ij = colors[get_idx(i, j, N)]
                for k in range(i + 1, N):
                    # Triangle (i, j, k) with i < j < k
                    # Edges: (i,j), (i,k), (j,k)
                    if c_ij == colors[get_idx(i, k, N)] == colors[get_idx(j, k, N)]:
                        return False
        return True

    # CSP Solver for Extension
    def try_extend(colors_prev, N_prev):
        # Returns list of colors for edges (N_prev, i) for i in 0..N_prev-1
        # or None if failed
        N_new = N_prev + 1
        # Precompute colors of K_N_prev
        # c[i][j] for i < j < N_prev
        
        # Domains for x[0..N_prev-1]
        domains = [set([0, 1, 2, 3]) for _ in range(N_prev)]
        
        # We will assign x[0], x[1], ...
        # But we use MRV, so we need to track unassigned variables.
        
        x = [None] * N_prev
        assigned = [False] * N_prev
        
        def get_color_prev(i, j):
            # i < j < N_prev
            return colors_prev[get_idx(i, j, N_prev)]

        def forward_check(k, c):
            # Assign x[k] = c
            # Check if this makes any future domain empty
            for j in range(k + 1, N_prev):
                if assigned[j]:
                    continue
                if c in domains[j]:
                    # If c(k,j) == c, then x[j] cannot be c
                    if get_color_prev(k, j) == c:
                        domains[j].remove(c)
                        if len(domains[j]) == 0:
                            return False
            return True

        def backtrack():
            # Select unassigned variable with smallest domain (MRV)
            min_size = 5
            var = -1
            for i in range(N_prev):
                if not assigned[i]:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        var = i
                        if min_size == 1:
                            break
            if var == -1:
                return True # All assigned
            
            # Try values in domains[var]
            # Order values by LCV (Least Constraining Value) - simplified: random
            values = list(domains[var])
            random.shuffle(values)
            
            for c in values:
                x[var] = c
                assigned[var] = True
                
                # Save domains for backtracking
                saved_domains = [set(d) for d in domains]
                
                if forward_check(var, c):
                    if backtrack():
                        return True
                
                # Restore domains
                for i in range(N_prev):
                    domains[i] = saved_domains[i]
                
                x[var] = None
                assigned[var] = False
                
            return False

        # Add timeout check inside backtrack?
        # For now, assume it's fast. If not, we might need a step limit.
        # Let's add a simple step counter
        steps = [0]
        limit = 10000
        
        def backtrack_limited():
            steps[0] += 1
            if steps[0] > limit:
                return False
            
            min_size = 5
            var = -1
            for i in range(N_prev):
                if not assigned[i]:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        var = i
                        if min_size == 1:
                            break
            if var == -1:
                return True
            
            values = list(domains[var])
            random.shuffle(values)
            
            for c in values:
                x[var] = c
                assigned[var] = True
                saved_domains = [set(d) for d in domains]
                
                if forward_check(var, c):
                    if backtrack_limited():
                        return True
                
                for i in range(N_prev):
                    domains[i] = saved_domains[i]
                
                x[var] = None
                assigned[var] = False
                
            return False

        # Try multiple random restarts
        for _ in range(10):
            random.seed()
            domains = [set([0, 1, 2, 3]) for _ in range(N_prev)]
            x = [None] * N_prev
            assigned = [False] * N_prev
            steps = [0]
            if backtrack_limited():
                return x
        return None

    # Simulated Annealing Fallback
    def sa_repair(colors_prev, N_prev):
        N_new = N_prev + 1
        # Initialize
        x = [random.randint(0, 3) for _ in range(N_prev)]
        colors_new = colors_prev + x # Wait, indices are different
        
        # We need a unified color list for K_N_new
        # Edges within K_N_prev are in colors_prev
        # Edges to new vertex N_prev are x[0..N_prev-1]
        
        # Build full color list for K_N_new
        # Size: N_new * (N_new - 1) / 2
        full_colors = [0] * (N_new * (N_new - 1) // 2)
        
        # Copy prev
        for i in range(N_prev):
            for j in range(i + 1, N_prev):
                full_colors[get_idx(i, j, N_new)] = colors_prev[get_idx(i, j, N_prev)]
        
        # Add new edges
        for i in range(N_prev):
            full_colors[get_idx(i, N_prev, N_new)] = x[i]
            
        # Objective: Count mono triangles
        def count_triangles(colors):
            count = 0
            for i in range(N_new):
                for j in range(i + 1, N_new):
                    c = colors[get_idx(i, j, N_new)]
                    for k in range(j + 1, N_new):
                        if colors[get_idx(i, k, N_new)] == c and colors[get_idx(j, k, N_new)] == c:
                            count += 1
            return count

        current_energy = count_triangles(full_colors)
        T = 1.0
        alpha = 0.99
        min_T = 0.01
        
        while T > min_T:
            # Pick random edge
            i = random.randint(0, N_new - 1)
            j = random.randint(0, N_new - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            idx = get_idx(i, j, N_new)
            old_c = full_colors[idx]
            new_c = random.randint(0, 3)
            if old_c == new_c: continue
            
            full_colors[idx] = new_c
            new_energy = count_triangles(full_colors)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_energy = new_energy
            else:
                full_colors[idx] = old_c
                
            T *= alpha
            
            if current_energy == 0:
                return full_colors
            
        return None

    # Main Loop
    best_N = N
    best_colors = current_colors
    
    while N < 51:
        if time.time() > deadline:
            break
            
        # Try CSP Extension
        ext = try_extend(current_colors, N)
        if ext is not None:
            # Build new color list
            new_colors = current_colors + ext
            # Note: The order of colors in the list must be lexicographic (i,j) for i<j in K_{N+1}
            # The existing current_colors are for pairs (i,j) with i<j<N.
            # The new pairs are (i, N) for i<N.
            # In lexicographic order, (i, N) comes after all (i, j) with j<N?
            # Let's check index order.
            # Pairs: (0,1), (0,2)...(0,N-1), (1,2)...
            # So (i, N) are at the end?
            # No.
            # (0,1), (0,2), ..., (0,N-1), (1,2), ..., (1,N-1), ..., (N-2, N-1)
            # The new vertex is N.
            # Pairs involving N: (0,N), (1,N), ..., (N-1,N).
            # In lexicographic order for K_{N+1}:
            # (0,1), (0,2), ..., (0,N), (1,2), ..., (1,N), ..., (N-1,N)
            # So the new colors are interleaved!
            
            # We must reconstruct the list properly.
            new_colors = [0] * ((N + 1) * N // 2)
            # Copy old
            for i in range(N):
                for j in range(i + 1, N):
                    new_colors[get_idx(i, j, N + 1)] = current_colors[get_idx(i, j, N)]
            # Add new
            for i in range(N):
                new_colors[get_idx(i, N, N + 1)] = ext[i]
                
            # Verify
            if is_valid_coloring(new_colors, N + 1):
                N += 1
                current_colors = new_colors
                best_N = N
                best_colors = new_colors
                continue
            else:
                # Should not happen if CSP is correct, but safety
                pass
        
        # Try SA Repair
        if time.time() > deadline - 5:
            break
        repaired = sa_repair(current_colors, N)
        if repaired is not None and is_valid_coloring(repaired, N + 1):
            N += 1
            current_colors = repaired
            best_N = N
            best_colors = repaired
        else:
            # Failed to extend
            # We could try restarting the whole search for this N with different random seeds
            # But for now, we break to save time for other N?
            # Actually, if we can't extend N to N+1, we can't get N+2.
            # So we break.
            break

    return {'n': best_N, 'colors': best_colors}

BET: {mechanism: "CSP extension with forward checking allows rapid growth from N=32 to N=50 by exploiting the structure of the existing coloring, while SA provides a fallback for difficult steps.", q: 0.9, delta: [0.5, 0.6]}
