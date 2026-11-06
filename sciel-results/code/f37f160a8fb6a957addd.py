import time
import random
import numpy as np

def construct():
    deadline = time.time() + 135  # Safety margin
    
    def greenwood_gleason_32():
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
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return 32, colors

    def simulate_annealing(n, time_limit):
        """
        Attempts to find a 4-coloring of K_n with no monochromatic triangles.
        Returns (n, colors_list) if successful, else None.
        """
        start_time = time.time()
        
        # Initialize random coloring
        # adj[i][j] = color of edge (i,j)
        adj = np.random.randint(0, 4, size=(n, n))
        # Make it symmetric
        adj = (adj + adj.T) // 2
        # Set diagonal to 0 (irrelevant)
        np.fill_diagonal(adj, 0)
        
        # Function to calculate total energy (number of mono triangles)
        def calc_energy(adj):
            energy = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = adj[i, j]
                    # Count k > j such that adj[i,k]==c and adj[j,k]==c
                    # Vectorized:
                    # mask_i = (adj[i, j+1:] == c)
                    # mask_j = (adj[j, j+1:] == c)
                    # count = np.sum(mask_i & mask_j)
                    # This is O(n^3) total if done naively for all i,j.
                    # Better: iterate triangles.
                    # For n=50, n^3/6 ~ 20k triangles. 
                    # We can just sum over all triples.
            # Actually, for initial energy, we can just count all mono triangles.
            # Efficient way:
            # For each color c, count triangles.
            # Or simply:
            energy = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = adj[i, j]
                    # k must be > j to avoid double counting? 
                    # No, triangle (i,j,k) is unique.
                    # If we sum over all i<j<k, we count each triangle once.
                    # mask = (adj[i, j+1:] == c) & (adj[j, j+1:] == c)
                    # This requires adj[i, j+1:] and adj[j, j+1:] to be aligned.
                    # adj[i, k] and adj[j, k] for k in j+1..n-1.
                    # Yes, this works.
                    energy += np.sum((adj[i, j+1:] == c) & (adj[j, j+1:] == c))
            return energy

        current_energy = calc_energy(adj)
        
        # If we start lucky
        if current_energy == 0:
            return adj_to_colors(adj, n)
            
        T = 1.0
        cooling = 0.999
        # We want to run for time_limit seconds
        # Estimate steps per second? Just loop until time is up.
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Ensure u < v for consistency, though adj is symmetric
            if u > v:
                u, v = v, u
                
            c_old = adj[u, v]
            # Pick new color
            # c_new != c_old
            # We can pick from 0..3 excluding c_old
            c_new = random.randint(0, 3)
            if c_new == c_old:
                c_new = (c_new + 1) % 4
            
            # Calculate Delta Energy
            # Triangles containing (u,v) are (u,v,w) for w != u,v
            # Old contribution: w such that adj[u,w]==c_old and adj[v,w]==c_old
            # New contribution: w such that adj[u,w]==c_new and adj[v,w]==c_new
            
            # We need to exclude u and v from w.
            # adj[u, :] is row u. adj[v, :] is row v.
            # mask_u_old = (adj[u, :] == c_old)
            # mask_v_old = (adj[v, :] == c_old)
            # count_old = np.sum(mask_u_old & mask_v_old)
            # This includes w=u and w=v.
            # For w=u: adj[u,u]=0. If c_old==0, it counts. But edge (u,u) doesn't exist.
            # We must exclude w=u and w=v.
            
            mask_u_old = (adj[u, :] == c_old)
            mask_v_old = (adj[v, :] == c_old)
            # Set indices u and v to False to exclude them
            mask_u_old[u] = False
            mask_u_old[v] = False
            mask_v_old[u] = False
            mask_v_old[v] = False
            
            count_old = np.sum(mask_u_old & mask_v_old)
            
            mask_u_new = (adj[u, :] == c_new)
            mask_v_new = (adj[v, :] == c_new)
            mask_u_new[u] = False
            mask_u_new[v] = False
            mask_v_new[u] = False
            mask_v_new[v] = False
            
            count_new = np.sum(mask_u_new & mask_v_new)
            
            delta_e = count_new - count_old
            
            # Acceptance
            if delta_e < 0 or (delta_e == 0 and random.random() < 0.5):
                adj[u, v] = c_new
                adj[v, u] = c_new
                current_energy += delta_e
                if current_energy == 0:
                    return adj_to_colors(adj, n)
            else:
                # Accept with probability exp(-delta_e / T)
                if random.random() < np.exp(-delta_e / T):
                    adj[u, v] = c_new
                    adj[v, u] = c_new
                    current_energy += delta_e
                    if current_energy == 0:
                        return adj_to_colors(adj, n)
            
            T *= cooling
            if T < 0.01:
                T = 0.01 # Keep it low but not zero, or restart?
                # Actually, if T is too low, we get stuck. 
                # We can reset T occasionally or use a different schedule.
                # For now, just let it cool. If stuck, we might need restarts.
                
        return None

    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(adj[i, j]))
        return colors

    def validate(colors, n):
        # Reconstruct adj
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        # Check mono triangles
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        return False
        return True

    # Strategy:
    # 1. Try N=50 with SA.
    # 2. If fails, try N=49, 48...
    # 3. Fallback to N=32.

    best_n = 32
    best_colors = greenwood_gleason_32()[1]
    
    # We have ~135s.
    # Allocate time for each N.
    # N=50 is the most important.
    
    # Try N=50
    time_for_50 = min(60, deadline - time.time() - 5)
    if time_for_50 > 5:
        res = simulate_annealing(50, time_for_50)
        if res is not None:
            if validate(res, 50):
                return {'n': 50, 'colors': res}
    
    # Try N=49
    time_for_49 = min(30, deadline - time.time() - 5)
    if time_for_49 > 5:
        res = simulate_annealing(49, time_for_49)
        if res is not None:
            if validate(res, 49):
                return {'n': 49, 'colors': res}
                
    # Try N=48
    time_for_48 = min(20, deadline - time.time() - 5)
    if time_for_48 > 5:
        res = simulate_annealing(48, time_for_48)
        if res is not None:
            if validate(res, 48):
                return {'n': 48, 'colors': res}

    # Fallback
    return {'n': best_n, 'colors': best_colors}
