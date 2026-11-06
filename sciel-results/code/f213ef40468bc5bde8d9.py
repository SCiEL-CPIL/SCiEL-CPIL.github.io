"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(42)
    
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
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
    
    def build_k16_colors():
        colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                colors.append(log[i ^ j] % 3)
        return colors
    
    def build_k32_colors():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors
    
    # Efficient incremental validity check using adjacency bitsets
    def make_adjacency(colors, n):
        """Build adjacency lists per colour for fast triangle checking."""
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        return adj
    
    def is_valid_incremental(base_colors, base_n, new_edges):
        """
        Check if adding new vertex (index base_n) with given edge colours
        to vertices 0..base_n-1 creates any monochromatic triangle.
        new_edges[i] = colour of edge (new_vertex, i)
        """
        # For each colour c, find vertices i where new_edges[i] == c
        # For each pair (i,j) both in that set, check if edge (i,j) has colour c
        for c in range(4):
            verts_c = []
            for i in range(base_n):
                if new_edges[i] == c:
                    verts_c.append(i)
            for a in range(len(verts_c)):
                i = verts_c[a]
                for b in range(a + 1, len(verts_c)):
                    j = verts_c[b]
                    # Get index of edge (i,j) in base_colors
                    idx = i * (2 * base_n - 1 - i) // 2 + (j - i - 1)
                    if base_colors[idx] == c:
                        return False
        return True
    
    def count_conflicts(base_colors, base_n, new_edges):
        """Count number of monochromatic triangles that would be created."""
        count = 0
        for c in range(4):
            verts_c = []
            for i in range(base_n):
                if new_edges[i] == c:
                    verts_c.append(i)
            for a in range(len(verts_c)):
                i = verts_c[a]
                for b in range(a + 1, len(verts_c)):
                    j = verts_c[b]
                    idx = i * (2 * base_n - 1 - i) // 2 + (j - i - 1)
                    if base_colors[idx] == c:
                        count += 1
        return count
    
    # Start with K32
    current_n = 32
    current_colors = build_k32_colors()
    best_n = 32
    best_colors = current_colors[:]
    
    # Precompute: for each pair (i,j) in current graph, store the colour
    # This allows O(1) lookup during extension
    
    def get_color(colors, n, i, j):
        """Get colour of edge (i,j) in K_n colouring."""
        if i > j:
            i, j = j, i
        idx = i * (2 * n - 1 - i) // 2 + (j - i - 1)
        return colors[idx]
    
    # Try to extend one vertex at a time
    # Use local search: start with random assignment, minimize conflicts
    max_target = 50
    improved = True
    
    while current_n < max_target and improved and time.time() < deadline - 10:
        improved = False
        base_n = current_n
        base_colors = current_colors
        
        # Precompute colour lookup for speed
        color_lookup = {}
        idx = 0
        for i in range(base_n):
            for j in range(i + 1, base_n):
                color_lookup[(i, j)] = base_colors[idx]
                idx += 1
        
        def get_color_fast(i, j):
            if i > j:
                i, j = j, i
            return color_lookup.get((i, j), -1)
        
        # Local search to find valid extension
        best_conflicts = None
        best_new_edges = None
        
        for restart in range(3000):
            if time.time() > deadline - 10:
                break
            
            # Random initial assignment
            new_edges = [random.randint(0, 3) for _ in range(base_n)]
            
            # Local search: flip colours to reduce conflicts
            for step in range(5000):
                if time.time() > deadline - 10:
                    break
                
                # Count conflicts
                conflicts = 0
                conflict_pairs = []
                for c in range(4):
                    verts_c = []
                    for i in range(base_n):
                        if new_edges[i] == c:
                            verts_c.append(i)
                    for a in range(len(verts_c)):
                        i = verts_c[a]
                        for b in range(a + 1, len(verts_c)):
                            j = verts_c[b]
                            if get_color_fast(i, j) == c:
                                conflicts += 1
                                conflict_pairs.append((i, j, c))
                
                if conflicts == 0:
                    if best_conflicts is None or conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_new_edges = new_edges[:]
                    improved = True
                    break
                
                if best_conflicts is None or conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_new_edges = new_edges[:]
                
                if not conflict_pairs:
                    break
                
                # Pick a random conflict and fix it
                i, j, c = random.choice(conflict_pairs)
                # Change colour of edge to new vertex for i or j
                if random.random() < 0.5:
                    new_edges[i] = (new_edges[i] + random.randint(1, 3)) % 4
                else:
                    new_edges[j] = (new_edges[j] + random.randint(1, 3)) % 4
            
            if improved:
                break
        
        if not improved:
            # Try more aggressive: simulated annealing style
            if time.time() > deadline - 15:
                break
            
            new_edges = [random.randint(0, 3) for _ in range(base_n)]
            current_conflicts = count_conflicts(base_colors, base_n, new_edges)
            best_conflicts_sa = current_conflicts
            best_new_edges_sa = new_edges[:]
            
            for step in range(20000):
                if time.time() > deadline - 10:
                    break
                
                # Pick random vertex and change its colour
                v = random.randint(0, base_n - 1)
                old_c = new_edges[v]
                new_c = (old_c + random.randint(1, 3)) % 4
                new_edges[v] = new_c
                
                new_conflicts = count_conflicts(base_colors, base_n, new_edges)
                
                if new_conflicts <= current_conflicts or random.random() < 0.1:
                    current_conflicts = new_conflicts
                    if new_conflicts < best_conflicts_sa:
                        best_conflicts_sa = new_conflicts
                        best_new_edges_sa = new_edges[:]
                else:
                    new_edges[v] = old_c
                
                if best_conflicts_sa == 0:
                    improved = True
                    break
            
            if best_conflicts_sa == 0:
                improved = True
                new_edges = best_new_edges_sa
        
        if improved:
            current_colors.extend(best_new_edges)
            current_n += 1
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors[:]
        else:
            break
    
    # Final validation
    if best_n > 1:
        # Verify no monochromatic triangle
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(best_n):
            for j in range(i + 1, best_n):
                c = best_colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        valid = True
        for c in range(4):
            for (i, j) in adj[c]:
                for k in range(i):
                    if (k, i) in adj[c] and (k, j) in adj[c]:
                        valid = False
                        break
                if not valid:
                    break
                for k in range(j + 1, best_n):
                    if (i, k) in adj[c] and (j, k) in adj[c]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to K32
            best_n = 32
            best_colors = build_k32_colors()
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
