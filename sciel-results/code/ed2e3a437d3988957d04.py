import time
import random
import math

def construct():
    deadline = time.time() + 115

    # --- Base n=32 colouring (Greenwood-Gleason) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def edge_color_core(i, j):
        return log_map[i ^ j] % 3

    n_base = 32
    base_colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                c = edge_color_core(i % 16, j % 16)
            else:
                c = 3
            base_colors[idx] = c
            idx += 1

    # --- Block construction for n=50 ---
    # Partition 50 vertices into 4 blocks: sizes 13, 13, 12, 12
    # Inter-block edges: use a fixed 3-coloring of K_4 (no mono triangles)
    # Intra-block edges: optimize to avoid mono triangles

    # A valid 3-coloring of K_4 without monochromatic triangles:
    # Vertices 0,1,2,3
    # (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=2, (1,3)=1, (2,3)=0
    # Check: triangle (0,1,2): colors 0,1,2 - ok
    #         triangle (0,1,3): colors 0,2,1 - ok
    #         triangle (0,2,3): colors 1,2,0 - ok
    #         triangle (1,2,3): colors 2,1,0 - ok
    k4_coloring = {}
    k4_coloring[(0, 1)] = 0
    k4_coloring[(0, 2)] = 1
    k4_coloring[(0, 3)] = 2
    k4_coloring[(1, 2)] = 2
    k4_coloring[(1, 3)] = 1
    k4_coloring[(2, 3)] = 0

    def solve_block(n_block, start_vertex, deadline):
        """Solve a triangle-free 3-coloring of K_n_block using local search."""
        if n_block <= 1:
            return []
        
        num_edges = n_block * (n_block - 1) // 2
        edges = []
        edge_to_idx = {}
        for i in range(n_block):
            for j in range(i + 1, n_block):
                edge_to_idx[(i, j)] = len(edges)
                edges.append((i, j))
        
        # Precompute neighbors for each edge (triangles containing this edge)
        edge_neighbors = [[] for _ in range(num_edges)]
        for idx, (u, v) in enumerate(edges):
            for w in range(n_block):
                if w == u or w == v:
                    continue
                idx_uw = edge_to_idx[(min(u, w), max(u, w))]
                idx_vw = edge_to_idx[(min(v, w), max(v, w))]
                edge_neighbors[idx].append((w, idx_uw, idx_vw))
        
        # Greedy initialization: try to build a triangle-free coloring
        best_colors = None
        best_mono_count = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline - 1:
                break
            colors = [random.randint(0, 2) for _ in range(num_edges)]
            
            # Count monochromatic triangles
            mono_count = 0
            for i in range(n_block - 2):
                for j in range(i + 1, n_block - 1):
                    c_ij = colors[edge_to_idx[(i, j)]]
                    for k in range(j + 1, n_block):
                        if colors[edge_to_idx[(j, k)]] == c_ij and colors[edge_to_idx[(i, k)]] == c_ij:
                            mono_count += 1
            
            if mono_count == 0:
                return colors
            if mono_count < best_mono_count:
                best_mono_count = mono_count
                best_colors = colors[:]
            
            # Local search to reduce mono triangles
            temperature = 1.0
            for step in range(20000):
                if time.time() > deadline - 1:
                    break
                if mono_count == 0:
                    return colors
                
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                u, v = edges[idx]
                old_c = colors[idx]
                
                # Try all 3 colors
                best_delta = float('inf')
                best_new_c = -1
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    created = 0
                    destroyed = 0
                    for w, idx_uw, idx_vw in edge_neighbors[idx]:
                        c_uw = colors[idx_uw]
                        c_vw = colors[idx_vw]
                        was_mono = (old_c == c_uw == c_vw)
                        is_mono_new = (new_c == c_uw == c_vw)
                        if was_mono:
                            destroyed += 1
                        if is_mono_new:
                            created += 1
                    delta = created - destroyed
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                
                if best_new_c == -1:
                    continue
                
                accept = False
                if best_delta < 0:
                    accept = True
                elif best_delta == 0:
                    accept = random.random() < 0.3
                else:
                    accept = random.random() < math.exp(-best_delta / temperature)
                
                if accept:
                    colors[idx] = best_new_c
                    mono_count += best_delta
                    temperature *= 0.999
                    if temperature < 0.01:
                        temperature = 0.01
        
        return best_colors if best_colors is not None else [0] * num_edges

    # Try to build n=50 coloring using block construction
    N = 50
    block_sizes = [13, 13, 12, 12]
    num_blocks = 4
    
    # Assign vertices to blocks
    vertex_block = []
    for b, size in enumerate(block_sizes):
        for _ in range(size):
            vertex_block.append(b)
    
    # Build edge list
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
    num_edges = len(edges)
    
    # Initialize colors: inter-block edges use K4 coloring, intra-block edges start random
    colors = [0] * num_edges
    for idx, (u, v) in enumerate(edges):
        bu = vertex_block[u]
        bv = vertex_block[v]
        if bu != bv:
            # Use K4 coloring for inter-block edges
            colors[idx] = k4_coloring[(min(bu, bv), max(bu, bv))]
        else:
            colors[idx] = random.randint(0, 2)
    
    # Optimize intra-block colorings
    # For each block, extract the subproblem and solve it
    for b in range(num_blocks):
        if time.time() > deadline - 5:
            break
        block_vertices = [v for v in range(N) if vertex_block[v] == b]
        n_b = len(block_vertices)
        
        # Get current intra-block colors
        block_edges = []
        for i in range(n_b):
            for j in range(i + 1, n_b):
                vi = block_vertices[i]
                vj = block_vertices[j]
                e_idx = edge_to_idx[(min(vi, vj), max(vi, vj))]
                block_edges.append((i, j, e_idx))
        
        # Solve the block
        block_solution = solve_block(n_b, 0, deadline)
        
        # Apply solution
        for (i, j, e_idx), c in zip(block_edges, block_solution):
            colors[e_idx] = c
    
    # Verify no monochromatic triangles
    def count_mono_triangles(colors):
        count = 0
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                c_ij = colors[edge_to_idx[(i, j)]]
                for k in range(j + 1, N):
                    if colors[edge_to_idx[(j, k)]] == c_ij and colors[edge_to_idx[(i, k)]] == c_ij:
                        count += 1
        return count
    
    # If the block construction didn't yield a valid coloring, fall back to full local search
    mono_count = count_mono_triangles(colors)
    
    if mono_count > 0 and time.time() < deadline - 10:
        # Full local search on the whole graph
        # Precompute neighbors
        edge_neighbors = [[] for _ in range(num_edges)]
        for idx, (u, v) in enumerate(edges):
            for w in range(N):
                if w == u or w == v:
                    continue
                idx_uw = edge_to_idx[(min(u, w), max(u, w))]
                idx_vw = edge_to_idx[(min(v, w), max(v, w))]
                edge_neighbors[idx].append((w, idx_uw, idx_vw))
        
        # Simulated annealing
        temperature = 2.0
        for step in range(200000):
            if time.time() > deadline - 2:
                break
            if mono_count == 0:
                break
            
            idx = random.randint(0, num_edges - 1)
            old_c = colors[idx]
            
            best_delta = float('inf')
            best_new_c = -1
            for new_c in range(4):
                if new_c == old_c:
                    continue
                created = 0
                destroyed = 0
                for w, idx_uw, idx_vw in edge_neighbors[idx]:
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    was_mono = (old_c == c_uw == c_vw)
                    is_mono_new = (new_c == c_uw == c_vw)
                    if was_mono:
                        destroyed += 1
                    if is_mono_new:
                        created += 1
                delta = created - destroyed
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_new_c == -1:
                continue
            
            accept = False
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = random.random() < 0.3
            else:
                accept = random.random() < math.exp(-best_delta / temperature)
            
            if accept:
                colors[idx] = best_new_c
                mono_count += best_delta
                temperature *= 0.9995
                if temperature < 0.01:
                    temperature = 0.01
        
        mono_count = count_mono_triangles(colors)
    
    if mono_count == 0:
        return {'n': N, 'colors': colors}
    
    return {'n': n_base, 'colors': base_colors}
