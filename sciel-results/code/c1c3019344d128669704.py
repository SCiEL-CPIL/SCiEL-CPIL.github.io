import time
import random
import math

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    # --- GF(16) arithmetic for the seed coloring ---
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
    
    # --- Seed: n=32 Greenwood-Gleason coloring ---
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def colors_to_adj(colors, n):
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj[i][j])
        return colors
    
    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count
    
    # --- Core: Fast Simulated Annealing with delta-energy ---
    def simulated_annealing(adj, n, time_limit, start_temp=5.0):
        """Minimize monochromatic triangles. Returns (adj, energy)."""
        start_time = time.time()
        temp = start_temp
        # Geometric cooling tuned for ~1e5-1e6 steps
        cooling = 0.9995 
        best_energy = count_mono_triangles(adj, n)
        best_adj = [row[:] for row in adj]
        current_energy = best_energy
        
        # Precompute neighbor lists for each edge? No, n is small enough to just loop.
        # But we can optimize the delta calculation slightly by using local variables.
        
        while time.time() - start_time < time_limit:
            if temp < 0.01:
                break
            
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if j < i:
                i, j = j, i
            
            old_c = adj[i][j]
            new_c = random.choice([0, 1, 2, 3])
            if new_c == old_c:
                continue
            
            # Compute delta in monochromatic triangles
            # A triangle (i, j, k) is mono if adj[i][j]==adj[i][k]==adj[j][k]
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                # Removing old color
                if adj[i][k] == old_c and adj[j][k] == old_c:
                    delta -= 1
                # Adding new color
                if adj[i][k] == new_c and adj[j][k] == new_c:
                    delta += 1
            
            if delta <= 0:
                # Accept
                adj[i][j] = adj[j][i] = new_c
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = [row[:] for row in adj]
                    if best_energy == 0:
                        break
            else:
                # Reject or accept with prob exp(-delta/temp)
                # Optimization: if delta is large and temp is small, skip exp
                if delta > 50 * temp:
                    continue
                if random.random() < math.exp(-delta / temp):
                    adj[i][j] = adj[j][i] = new_c
                    current_energy += delta
            
            temp *= cooling
        
        return best_adj, best_energy
    
    # --- Core: Backtracking to add a vertex ---
    def try_add_vertex(adj, n, time_limit=5.0):
        """
        Attempts to find a valid coloring for edges (new_vertex, v) for all v in 0..n-1
        such that no monochromatic triangle is formed.
        Returns list of colors for the new vertex or None.
        """
        start_time = time.time()
        assign = [-1] * n
        
        # Order vertices by degree of constraint? 
        # Actually, just process 0..n-1.
        # Constraint: For each v, the color c assigned to (new, v) must not equal adj[u][v] 
        # for any u where assign[u] == c.
        
        # Let's use a recursive backtracking with forward checking.
        
        def is_valid(v, c):
            # Check if assigning color c to edge (new, v) creates a triangle
            # Triangle (new, v, u) is mono if adj[v][u] == c AND assign[u] == c
            for u in range(v):
                if assign[u] == c and adj[v][u] == c:
                    return False
            return True
        
        def bt(v):
            if time.time() - start_time > time_limit:
                return False
            if v == n:
                return True
            
            # Try colors 0, 1, 2, 3
            # Heuristic: try colors that appear less frequently in assign[0..v-1]?
            # Or just standard order.
            for c in range(4):
                if is_valid(v, c):
                    assign[v] = c
                    if bt(v + 1):
                        return True
                    assign[v] = -1
            return False
        
        # Randomize vertex order? No, fixed order is fine for determinism, 
        # but we can shuffle the *colors* tried or the *vertices*?
        # Shuffling vertices is risky for correctness if we don't track indices carefully.
        # Let's stick to fixed vertex order 0..n-1 but maybe shuffle color order?
        # Actually, for speed, fixed order is often fine.
        
        # Optimization: Pre-check if any vertex has no available color?
        # That's complex. Just run BT.
        
        # To avoid getting stuck in the same dead-end repeatedly if called multiple times,
        # we rely on the external loop to change the base coloring.
        
        if bt(0):
            return assign
        return None
    
    # --- Strategy ---
    # 1. Start with n=32 seed.
    # 2. Loop:
    #    a. Try to add a vertex via backtracking.
    #    b. If success, continue.
    #    c. If fail, run SA on the current graph to "relax" it, then retry.
    #    d. If SA doesn't help after some tries, or if we hit time limit, stop.
    # 3. Also, periodically try to start SA from scratch on a larger n (e.g., 40, 45) 
    #    seeded with the current best or random, to see if we can find a 0-energy state directly.
    
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Phase 1: Iterative extension
    # We will try to extend from 32 upwards.
    # If we get stuck, we run SA on the current n to try to find a "flexible" coloring
    # that allows extension.
    
    stuck_count = 0
    max_stuck = 10 # If we fail 10 times in a row to extend, try a different approach
    
    while time.time() < deadline - 20:
        n = len(adj)
        
        # Try to add a vertex
        # Use a small time limit for backtracking since it's exponential
        bt_limit = min(5.0, (deadline - time.time()) / 10)
        new_vertex_colors = try_add_vertex(adj, n, time_limit=bt_limit)
        
        if new_vertex_colors is not None:
            # Success! Add the vertex
            new_n = n + 1
            new_adj = [row[:] for row in adj]
            for _ in range(new_n):
                new_adj.append([-1] * new_n)
            for i in range(n):
                new_adj[i][n] = new_adj[n][i] = new_vertex_colors[i]
            new_adj[n][n] = -1
            
            adj = new_adj
            best_n = new_n
            best_colors = adj_to_colors(adj, best_n)
            stuck_count = 0
            # Update best
        else:
            # Failed to extend
            stuck_count += 1
            
            if stuck_count > max_stuck:
                # We are stuck. Try SA on the current graph to see if we can find
                # a different configuration that allows extension.
                # Or, try SA on a larger graph directly?
                
                # Option A: Relax current graph
                sa_limit = min(10.0, (deadline - time.time()) / 5)
                if sa_limit > 2.0:
                    adj_sa, energy = simulated_annealing(adj, n, sa_limit)
                    if energy < count_mono_triangles(adj, n):
                        adj = adj_sa
                        # Reset stuck count, maybe this new config allows extension
                        stuck_count = 0
                    else:
                        # SA didn't improve, maybe we can't extend this specific graph.
                        # Try a "jump": randomize a few edges to break symmetry?
                        # Or just give up on this path and try a fresh start for larger n?
                        pass
                else:
                    break
                
                # If we've been stuck for a while, try to initialize a larger n from scratch
                # using the current best as a seed for the first n vertices?
                # This is complex. Let's just keep relaxing.
                
            # If stuck_count is high, maybe we should try to "shake" the graph
            # by doing a few random valid moves (delta=0 moves) to escape local minima
            # in the space of valid colorings.
            if stuck_count > 2:
                # Perform random walks in the space of valid colorings (0-energy)
                # If current graph has 0 triangles, we can move edges that don't create triangles.
                # If it has >0 triangles, we should probably run SA first.
                current_energy = count_mono_triangles(adj, n)
                if current_energy == 0:
                    # Try to find a different 0-energy coloring that is more "extendable"
                    # Simple random walk: pick edge, if changing it keeps energy 0, do it.
                    for _ in range(100):
                        i = random.randint(0, n-1)
                        j = random.randint(0, n-1)
                        if i == j: continue
                        if j < i: i, j = j, i
                        old_c = adj[i][j]
                        new_c = random.choice([0, 1, 2, 3])
                        if new_c == old_c: continue
                        
                        delta = 0
                        for k in range(n):
                            if k == i or k == j: continue
                            if adj[i][k] == old_c and adj[j][k] == old_c: delta -= 1
                            if adj[i][k] == new_c and adj[j][k] == new_c: delta += 1
                        
                        if delta == 0:
                            adj[i][j] = adj[j][i] = new_c
                else:
                    # Has triangles, run SA
                    sa_limit = min(5.0, (deadline - time.time()) / 5)
                    if sa_limit > 1.0:
                        adj, energy = simulated_annealing(adj, n, sa_limit)
                        stuck_count = 0 # Reset because we changed the graph significantly
    
    # Phase 2: If we didn't reach high n, try direct SA on larger n
    # This is a fallback. If we are at n=32, try to find a 50-vertex solution directly.
    # This is unlikely to work from random, but maybe from seed?
    # Given the time, this is a long shot. The iterative extension is the main hope.
    
    # Final verification
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        # Fallback to seed
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
