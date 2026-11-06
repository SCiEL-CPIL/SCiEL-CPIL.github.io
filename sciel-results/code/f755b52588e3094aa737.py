import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Precompute all C6 cycles for a given n
    # A C6 is a simple cycle of 6 vertices.
    # We represent each cycle as a tuple of 6 vertices (v0, v1, v2, v3, v4, v5)
    # such that v0 is the smallest vertex, and v1 < v5 (to fix orientation).
    # We also build a mapping from edge (u,v) to list of cycle indices containing that edge.
    
    def precompute_cycles(n):
        cycles = []
        edge_to_cycles = {}
        
        # Generate all combinations of 6 vertices
        for verts in combinations(range(n), 6):
            v0 = verts[0]
            rest = verts[1:]
            
            # Generate all permutations of the remaining 5 vertices
            # To avoid duplicates (reversals), we only keep permutations where perm[0] < perm[-1]
            for perm in permutations(rest):
                if perm[0] < perm[-1]:
                    cycle = (v0, perm[0], perm[1], perm[2], perm[3], perm[4])
                    idx = len(cycles)
                    cycles.append(cycle)
                    
                    # Add edges to the mapping
                    edges = [
                        (cycle[0], cycle[1]),
                        (cycle[1], cycle[2]),
                        (cycle[2], cycle[3]),
                        (cycle[3], cycle[4]),
                        (cycle[4], cycle[5]),
                        (cycle[5], cycle[0])
                    ]
                    for e in edges:
                        key = (min(e[0], e[1]), max(e[0], e[1]))
                        if key not in edge_to_cycles:
                            edge_to_cycles[key] = []
                        edge_to_cycles[key].append(idx)
        
        return cycles, edge_to_cycles

    def has_path_len_5(adj_c, u, v):
        """Check if there is a simple path of length 5 from u to v in graph adj_c."""
        def dfs(current, depth, visited):
            if depth == 5:
                return current == v
            if current == v:
                return False
            
            neighbors = adj_c[current] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    def greedy_solve(n, cycles, edge_to_cycles):
        """
        Greedy construction with backtracking-lite.
        We assign colors to edges in lexicographic order.
        For each edge, we try colors 0..3. A color is valid if it doesn't create a monochromatic C6.
        Checking for a monochromatic C6 after adding edge (u,v) with color c:
        We look at all cycles in edge_to_cycles[(u,v)]. For each such cycle, check if all other 5 edges
        already have color c. If so, adding this edge creates a monochromatic C6.
        """
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        # Edge list in lexicographic order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for i, j in edges:
            found = False
            # Shuffle colors for diversity
            color_order = [0, 1, 2, 3]
            random.shuffle(color_order)
            
            for c in color_order:
                valid = True
                # Check all cycles containing edge (i,j)
                key = (min(i, j), max(i, j))
                if key in edge_to_cycles:
                    for cyc_idx in edge_to_cycles[key]:
                        cycle = cycles[cyc_idx]
                        # Check if all other 5 edges in this cycle have color c
                        # The cycle is (v0, v1, v2, v3, v4, v5)
                        # Edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
                        # We are adding (i,j). The other 5 edges must be checked.
                        # We don't know which edge in the cycle is (i,j), so we check all 6,
                        # but for the one we are adding, we assume it's c.
                        # So we check if the other 5 are c.
                        
                        # To be efficient, we can just check if the count of edges with color c
                        # in the cycle (excluding the one we are adding) is 5.
                        # But we need to know which one we are adding.
                        
                        # Simpler: iterate through the 6 edges of the cycle.
                        # If the edge is (i,j), skip it (we assume it's c).
                        # Otherwise, check if colors[u][v] == c.
                        
                        all_other_c = True
                        for k in range(6):
                            u = cycle[k]
                            v = cycle[(k + 1) % 6]
                            if (u, v) == (i, j) or (u, v) == (j, i):
                                continue
                            if colors[u][v] != c:
                                all_other_c = False
                                break
                        
                        if all_other_c:
                            valid = False
                            break
                
                if valid:
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    colors[i][j] = c
                    colors[j][i] = c
                    found = True
                    break
            
            if not found:
                return None
        
        return colors

    def solve(n, cycles, edge_to_cycles):
        """
        Backtracking solver.
        """
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == len(edges):
                return colors
            
            i, j = edges[idx]
            
            # Symmetry breaking for first few edges
            if i == 0 and j == 1:
                color_range = [0]
            elif i == 0 and j == 2:
                color_range = [0, 1]
            else:
                color_range = [0, 1, 2, 3]
            
            for c in color_range:
                valid = True
                key = (min(i, j), max(i, j))
                if key in edge_to_cycles:
                    for cyc_idx in edge_to_cycles[key]:
                        cycle = cycles[cyc_idx]
                        all_other_c = True
                        for k in range(6):
                            u = cycle[k]
                            v = cycle[(k + 1) % 6]
                            if (u, v) == (i, j) or (u, v) == (j, i):
                                continue
                            if colors[u][v] != c:
                                all_other_c = False
                                break
                        
                        if all_other_c:
                            valid = False
                            break
                
                if valid:
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    colors[i][j] = c
                    colors[j][i] = c
                    
                    res = backtrack(idx + 1)
                    if res:
                        return res
                    
                    adj[c][i] &= ~(1 << j)
                    adj[c][j] &= ~(1 << i)
                    colors[i][j] = -1
                    colors[j][i] = -1
            
            return None
        
        return backtrack(0)

    # Strategy:
    # 1. Try n=17 with greedy.
    # 2. Try n=16 with greedy.
    # 3. Try n=15 with backtracking (faster than greedy for smaller n if greedy fails).
    # 4. Fallback to smaller n.

    # Precomputation time is significant.
    # For n=17, cycles ~ 742k. Precomputation might take 1-2s.
    # For n=16, cycles ~ 300k.
    
    # We will precompute for the largest n we try first, then reuse if possible?
    # No, cycles depend on n.
    
    # Let's try n=17 first.
    for n in [17, 16, 15]:
        if time.time() > deadline - 10:
            break
        
        # Precompute cycles
        t0 = time.time()
        cycles, edge_to_cycles = precompute_cycles(n)
        t1 = time.time()
        
        if t1 - t0 > 5:
            # Precomputation too slow, skip this n
            continue
            
        # Try greedy
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            res = greedy_solve(n, cycles, edge_to_cycles)
            if res:
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(res[i][j])
                return {'n': n, 'colors': flat_colors}
        
        # Try backtracking for n=15 (might be feasible)
        if n == 15:
            if time.time() > deadline - 5:
                break
            res = solve(n, cycles, edge_to_cycles)
            if res:
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(res[i][j])
                return {'n': n, 'colors': flat_colors}
    
    # Fallback to n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
