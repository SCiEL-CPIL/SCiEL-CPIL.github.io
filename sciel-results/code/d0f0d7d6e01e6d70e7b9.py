import time
import random
import math

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    # GF(16) multiplication for Greenwood-Gleason construction
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
    
    # Generate n=32 Greenwood-Gleason coloring
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
    
    def build_bitmasks(adj, n):
        """Build bitmasks: for each vertex v and color c, mask of neighbors u where adj[v][u]==c"""
        masks = [[0]*4 for _ in range(n)]
        for v in range(n):
            for u in range(n):
                if u != v:
                    c = adj[v][u]
                    masks[v][c] |= (1 << u)
        return masks
    
    def count_mono_triangles_fast(masks, n):
        """Count monochromatic triangles using bitmasks"""
        count = 0
        for i in range(n):
            for c in range(4):
                # For color c, look at neighbors of i with color c
                neighbors = masks[i][c]
                # For each pair (j,k) in neighbors, check if adj[j][k]==c
                # Use bit operations: for each j in neighbors, count neighbors of j with color c that are also in neighbors
                j = 0
                while neighbors:
                    if neighbors & 1:
                        # j is in neighbors
                        common = masks[j][c] & neighbors
                        count += bin(common).count('1')
                    neighbors >>= 1
                    j += 1
        return count // 3  # each triangle counted 3 times
    
    def try_add_vertex_fast(adj, n, time_limit=3.0):
        """
        Try to add vertex n to existing coloring of n vertices.
        Uses greedy + local search with bitmask-based conflict counting.
        """
        start_time = time.time()
        
        # Precompute masks for existing vertices
        masks = build_bitmasks(adj, n)
        
        # For each color c and vertex v, we need:
        # If new vertex gets color c to vertex v, then for any u where adj[v][u]==c and new vertex also gets color c to u,
        # we have a triangle.
        # So for each color c, the set of vertices assigned color c must be an independent set in G_c.
        
        # Greedy initialization: assign each vertex a color that minimizes immediate conflicts
        assignment = [0]*n
        for v in range(n):
            best_c = 0
            best_conf = -1
            for c in range(4):
                # If we assign color c to v, conflicts are:
                # For each u already assigned color c, if adj[v][u]==c, that's a conflict
                conf = 0
                for u in range(v):
                    if assignment[u] == c and adj[v][u] == c:
                        conf += 1
                if conf < best_conf or best_conf == -1:
                    best_conf = conf
                    best_c = c
            assignment[v] = best_c
        
        def get_energy(assign):
            energy = 0
            for u in range(n):
                cu = assign[u]
                for v in range(u+1, n):
                    if assign[v] == cu and adj[u][v] == cu:
                        energy += 1
            return energy
        
        current_energy = get_energy(assignment)
        
        if current_energy == 0:
            return assignment
        
        best_assignment = assignment[:]
        best_energy = current_energy
        
        T = 5.0
        T_min = 0.001
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
            
            # Pick random vertex
            u = random.randint(0, n - 1)
            old_color = assignment[u]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Calculate delta energy efficiently
            delta = 0
            for v in range(n):
                if v == u:
                    continue
                cv = assignment[v]
                if old_color == cv and adj[u][v] == old_color:
                    delta -= 1
                if new_color == cv and adj[u][v] == new_color:
                    delta += 1
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                assignment[u] = new_color
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment[:]
                    if best_energy == 0:
                        return best_assignment
            
            T *= cooling
            
        return best_assignment if best_energy == 0 else None
    
    # Start with n=32
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Try to expand vertex by vertex
    for target_n in range(33, 51):
        if time.time() > deadline - 15:
            break
        
        n = best_n
        found = False
        
        # Multiple attempts with different strategies
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            random.seed(1000 + target_n * 10 + attempt)
            assignment = try_add_vertex_fast(adj, n, time_limit=2.5)
            if assignment is not None:
                # Success! Add vertex
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
                break
        
        if not found:
            break
    
    # Verify the result
    if best_n > 32:
        adj_final = colors_to_adj(best_colors, best_n)
        masks_final = build_bitmasks(adj_final, best_n)
        if count_mono_triangles_fast(masks_final, best_n) > 0:
            return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
