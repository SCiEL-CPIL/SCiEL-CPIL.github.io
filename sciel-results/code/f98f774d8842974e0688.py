import time
import itertools
import random

def construct():
    t0 = time.time()
    deadline = 145  # safety margin before 150s kill
    
    n = 16
    # Index for edge (i,j) in flat list, 0<=i<j<16
    # Number of edges before i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    # Then j-i-1 more
    def idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all K4s and their 6 edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append([idx(x, y) for x, y in edges])
    
    # Count bad K4s: K4s with fewer than 5 distinct colors
    def count_bad(colors):
        bad = 0
        for edge_idxs in k4s:
            c = [colors[i] for i in edge_idxs]
            if len(set(c)) < 5:
                bad += 1
        return bad
    
    # Make a cyclic 1-factorization of K16
    # Vertices 0..14 on a cycle, 15 is "infinity"
    def make_1factorization_cyclic():
        colors = [0] * 120
        for factor in range(15):
            # Edge (15, factor)
            colors[idx(15, factor)] = factor
            for j in range(1, 8):
                a = (factor + j) % 15
                b = (factor - j) % 15
                if a > b:
                    a, b = b, a
                colors[idx(a, b)] = factor
        return colors
    
    # Get all edges in a given factor
    def get_factor_edges(colors, factor):
        edges = []
        for i in range(120):
            if colors[i] == factor:
                edges.append(i)
        return edges
    
    # Swap two edges between two factors
    def swap_edges(colors, e1, e2, f1, f2):
        """Swap colors of edges e1 and e2, ensuring they're in factors f1 and f2"""
        colors[e1] = f2
        colors[e2] = f1
    
    # Check if swapping maintains 1-factorization (each factor still has 8 edges, 
    # and no vertex has two edges of same color)
    def is_valid_swap(colors, e1, e2, f1, f2):
        # Get the vertices of e1 and e2
        def edge_vertices(eidx):
            # Reverse the idx computation
            # eidx = i*(n-1) - i*(i-1)//2 + (j-i-1)
            # This is complex, let's precompute
            return edge_verts[eidx]
        
        v1 = edge_vertices(e1)
        v2 = edge_vertices(e2)
        
        # Check that e1 is currently in f1 and e2 in f2
        if colors[e1] != f1 or colors[e2] != f2:
            return False
        
        # After swap: e1 gets f2, e2 gets f1
        # Check no vertex has two edges of same color
        # For each vertex in v1, check it doesn't have another edge of color f2
        # For each vertex in v2, check it doesn't have another edge of color f1
        for v in v1:
            for e in vertex_edges[v]:
                if e != e1 and colors[e] == f2:
                    return False
        for v in v2:
            for e in vertex_edges[v]:
                if e != e2 and colors[e] == f1:
                    return False
        return True
    
    # Precompute edge vertices and vertex edges
    edge_verts = []
    vertex_edges = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            e = idx(i, j)
            edge_verts.append((i, j))
            vertex_edges[i].append(e)
            vertex_edges[j].append(e)
    
    # Local search: try to reduce bad K4 count
    def local_search(colors, max_time=5):
        t_start = time.time()
        bad_count = count_bad(colors)
        if bad_count == 0:
            return colors, bad_count
        
        # Build list of edges per factor
        factor_edges = [[] for _ in range(15)]
        for e in range(120):
            factor_edges[colors[e]].append(e)
        
        # Try random swaps
        for iteration in range(100000):
            if time.time() - t_start > max_time:
                break
            if bad_count == 0:
                break
            
            # Pick two random factors
            f1, f2 = random.sample(range(15), 2)
            if not factor_edges[f1] or not factor_edges[f2]:
                continue
            
            e1 = random.choice(factor_edges[f1])
            e2 = random.choice(factor_edges[f2])
            
            if not is_valid_swap(colors, e1, e2, f1, f2):
                continue
            
            # Try the swap
            old_c1, old_c2 = colors[e1], colors[e2]
            colors[e1] = f2
            colors[e2] = f1
            
            new_bad = count_bad(colors)
            
            if new_bad <= bad_count:
                bad_count = new_bad
                factor_edges[f1].remove(e1)
                factor_edges[f2].remove(e2)
                factor_edges[f1].append(e2)
                factor_edges[f2].append(e1)
            else:
                # Revert
                colors[e1] = old_c1
                colors[e2] = old_c2
        
        return colors, bad_count
    
    # Fallback: (i+j) % 17
    def make_17coloring():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Try multiple 1-factorizations with local search
    best_colors = None
    best_bad = float('inf')
    
    # Try the standard cyclic 1-factorization
    colors = make_1factorization_cyclic()
    colors, bad = local_search(colors, max_time=30)
    if bad < best_bad:
        best_bad = bad
        best_colors = colors[:]
    
    if best_bad == 0:
        # Verify and return
        if count_bad(best_colors) == 0:
            return best_colors
    
    # Try a few more random 1-factorizations
    for attempt in range(5):
        if time.time() - t0 > deadline - 20:
            break
        
        # Generate a random 1-factorization by randomly permuting the cyclic one
        # or by random swaps
        colors = make_1factorization_cyclic()
        
        # Apply random valid swaps to perturb
        factor_edges = [[] for _ in range(15)]
        for e in range(120):
            factor_edges[colors[e]].append(e)
        
        for _ in range(50):
            f1, f2 = random.sample(range(15), 2)
            if not factor_edges[f1] or not factor_edges[f2]:
                continue
            e1 = random.choice(factor_edges[f1])
            e2 = random.choice(factor_edges[f2])
            if is_valid_swap(colors, e1, e2, f1, f2):
                colors[e1] = f2
                colors[e2] = f1
                factor_edges[f1].remove(e1)
                factor_edges[f2].remove(e2)
                factor_edges[f1].append(e2)
                factor_edges[f2].append(e1)
        
        colors, bad = local_search(colors, max_time=15)
        if bad < best_bad:
            best_bad = bad
            best_colors = colors[:]
        
        if best_bad == 0:
            break
    
    if best_colors is not None and best_bad == 0:
        if count_bad(best_colors) == 0:
            return best_colors
    
    # Fallback to 17-coloring
    return make_17coloring()
