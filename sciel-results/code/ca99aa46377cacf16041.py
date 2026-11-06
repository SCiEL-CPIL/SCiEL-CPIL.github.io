"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices.
    Uses blow-ups of C5 with local search on part sizes, plus Andrásfai graphs.
    Aims for n >= 27 vertices.
    """
    max_time = 100  # seconds
    start_time = time.time()
    
    def is_triangle_free(adj, n):
        """Check if graph is triangle-free using adjacency matrix."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def max_independent_set_size(adj, n):
        """
        Find the size of the maximum independent set using branch and bound.
        Returns the size.
        """
        best = [0]
        
        def branch(current_size, candidates):
            if time.time() - start_time > max_time:
                return
            if current_size + len(candidates) <= best[0]:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Pick vertex with highest degree in candidate subgraph for better pruning
            v = candidates[0]
            # Branch 1: include v
            new_candidates = [u for u in candidates[1:] if not adj[v][u]]
            branch(current_size + 1, new_candidates)
            # Branch 2: exclude v
            branch(current_size, candidates[1:])
        
        branch(0, list(range(n)))
        return best[0]
    
    def alpha_le_9(adj, n):
        """Check if independence number is at most 9."""
        if n <= 9:
            return True
        if n <= 30:
            return max_independent_set_size(adj, n) <= 9
        # For larger n, use greedy lower bound + exact if feasible
        used = [False] * n
        count = 0
        for v in range(n):
            if not used[v]:
                count += 1
                for u in range(n):
                    if adj[v][u]:
                        used[u] = True
        if count > 9:
            return False
        if n <= 35:
            return max_independent_set_size(adj, n) <= 9
        return True
    
    def c5_blowup(sizes):
        """
        Construct C5 blow-up with given part sizes.
        C5 vertices 0-4, edges: (0,1),(1,2),(2,3),(3,4),(4,0).
        Part i has sizes[i] vertices.
        Edges between part i and part j if i,j are adjacent in C5.
        """
        n = sum(sizes)
        # Assign vertex IDs to parts
        part_of = []
        for i, s in enumerate(sizes):
            part_of.extend([i] * s)
        
        adj = [[False] * n for _ in range(n)]
        # C5 adjacency: 0-1, 1-2, 2-3, 3-4, 4-0
        c5_adj = {(0,1), (1,0), (1,2), (2,1), (2,3), (3,2), (3,4), (4,3), (4,0), (0,4)}
        
        for u in range(n):
            for v in range(u+1, n):
                pu, pv = part_of[u], part_of[v]
                if (pu, pv) in c5_adj:
                    adj[u][v] = True
                    adj[v][u] = True
        
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        
        return {'n': n, 'edges': edges}
    
    def andrasfai_graph(k):
        """
        Construct Andrásfai graph And(2k+1) on 3(2k+1) vertices.
        Triangle-free with alpha = 2(2k+1).
        """
        m = 2 * k + 1  # odd number
        n = 3 * m
        # Vertices labeled (i, j) where i in 0..m-1, j in 0..2
        # Map to 0..n-1
        adj = [[False] * n for _ in range(n)]
        
        for i in range(m):
            for j in range(3):
                u = i * 3 + j
                # Connect to vertices in "next" layers based on Andrasfai construction
                # Standard Andrasfai: vertices (i,j), edges between (i,j) and (i',j')
                # if j' = j+1 mod 3 and i' = i + 1 + floor((j+j')/3) mod m
                for jp in range(3):
                    if jp == (j + 1) % 3:
                        ip = (i + 1 + (j + jp) // 3) % m
                        v = ip * 3 + jp
                        if u < v:
                            adj[u][v] = True
                            adj[v][u] = True
        
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        
        return {'n': n, 'edges': edges}
    
    def verify(graph):
        """Verify triangle-free and alpha <= 9."""
        n = graph['n']
        edges = graph['edges']
        adj = [[False] * n for _ in range(n)]
        for e in edges:
            adj[e[0]][e[1]] = True
            adj[e[1]][e[0]] = True
        
        if not is_triangle_free(adj, n):
            return False
        if n <= 35:
            alpha = max_independent_set_size(adj, n)
            if alpha > 9:
                return False
        return True
    
    # Strategy 1: Try Andrásfai graphs
    # And(9) on 27 vertices, alpha = 8
    result = andrasfai_graph(4)
    if verify(result):
        return result
    
    # Strategy 2: C5 blow-up with local search on sizes
    # Maximize sum(sizes) subject to:
    # s0+s2 <= 9, s0+s3 <= 9, s1+s3 <= 9, s1+s4 <= 9, s2+s4 <= 9, si >= 1
    best_n = 0
    best_result = None
    
    rng = random.Random(42)
    
    # Start with symmetric
    sizes = [4, 4, 4, 4, 4]  # n=20, alpha=8
    result = c5_blowup(sizes)
    if verify(result):
        best_n = result['n']
        best_result = result
    
    # Local search: try to increase sizes while maintaining constraints
    for iteration in range(10000):
        if time.time() - start_time > max_time - 10:
            break
        
        # Try to increase a random size by 1
        idx = rng.randint(0, 4)
        sizes_new = sizes[:]
        sizes_new[idx] += 1
        
        # Check constraints
        valid = True
        # Constraints: (0,2), (0,3), (1,3), (1,4), (2,4)
        constraints = [(0,2), (0,3), (1,3), (1,4), (2,4)]
        for a, b in constraints:
            if sizes_new[a] + sizes_new[b] > 9:
                valid = False
                break
        
        if valid:
            sizes = sizes_new
            result = c5_blowup(sizes)
            if verify(result):
                if result['n'] > best_n:
                    best_n = result['n']
                    best_result = result
    
    # Try to decrease some sizes and increase others
    for iteration in range(10000):
        if time.time() - start_time > max_time - 10:
            break
        
        idx_inc = rng.randint(0, 4)
        idx_dec = rng.randint(0, 4)
        if idx_inc == idx_dec:
            continue
        
        sizes_new = sizes[:]
        if sizes_new[idx_dec] <= 1:
            continue
        sizes_new[idx_inc] += 1
        sizes_new[idx_dec] -= 1
        
        valid = True
        constraints = [(0,2), (0,3), (1,3), (1,4), (2,4)]
        for a, b in constraints:
            if sizes_new[a] + sizes_new[b] > 9:
                valid = False
                break
        
        if valid:
            sizes = sizes_new
            result = c5_blowup(sizes)
            if verify(result):
                if result['n'] > best_n:
                    best_n = result['n']
                    best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return And(9)
    return andrasfai_graph(4)
# EVOLVE-BLOCK-END
