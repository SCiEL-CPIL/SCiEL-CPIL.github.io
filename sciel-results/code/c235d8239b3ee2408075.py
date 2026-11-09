"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Find a 4-coloring of K_n with no monochromatic C6, maximizing n.
    Uses backtracking with efficient C6 detection.
    """
    deadline = time.time() + 140.0  # 140s safety margin before 150s kill
    
    # We'll try to find colorings for n = 17, 16, 15, ...
    # Best known is 17. Let's try to find one.
    
    # Strategy: backtracking search with symmetry breaking
    # For n=17, we have C(17,2) = 136 edges to color with 4 colors
    
    def find_coloring(n, time_limit):
        """
        Backtracking search for a 4-coloring of K_n with no mono C6.
        Returns list of colors or None.
        """
        num_edges = n * (n - 1) // 2
        colors = [0] * num_edges
        
        # Edge index: for pair (i,j) with i<j, index = i*n - i*(i+1)//2 + (j - i - 1)
        # Actually, let's use a simpler mapping:
        # edge_index(i, j) where i < j
        # We can precompute this
        
        # Precompute edge indices
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        
        # For each color c, maintain adjacency sets
        # adj[c][v] = set of neighbors of v in color c
        adj = [set() for _ in range(n)]  # This won't work, need per color
        # adj[c][v] = set of u such that edge (min(u,v), max(u,v)) has color c
        adj = [dict() for _ in range(4)]
        for c in range(4):
            for v in range(n):
                adj[c][v] = set()
        
        # To check if adding edge (u,v) with color c creates a C6 in color c:
        # We need to check if there's a path of length 4 from u to v in color c
        # i.e., u - a - b - d - v where all edges are color c
        # This is equivalent to: exists a in N_c(u), d in N_c(v), such that
        # there exists b with b in N_c(a) and b in N_c(d)
        # Or more efficiently: for each a in N_c(u), for each d in N_c(v),
        # check if N_c(a) intersects N_c(d)
        
        # For efficiency, precompute for each color and each vertex, the set of neighbors
        # When adding edge (u,v) color c, check:
        # for a in adj[c][u]:
        #     for d in adj[c][v]:
        #         if adj[c][a] & adj[c][d]:  # intersection non-empty
        #             return False  # C6 found
        
        # But this is O(deg(u) * deg(v) * deg) which could be expensive.
        # Alternative: for each a in adj[c][u], for each d in adj[c][v],
        # check if there's a common neighbor. We can precompute common neighbors.
        
        # Actually, let's think differently. A C6 is u-a-b-d-v-w-u or similar.
        # Wait, a C6 containing edge (u,v) would be u-v-d-b-a-e-u or u-a-b-d-v-e-u.
        # If we're adding edge (u,v) with color c, a C6 would require a path of length 5 from u to v
        # in color c, OR a path of length 4 from u to some w and then w-v... no.
        
        # Let me reconsider. A cycle of length 6 containing edge (u,v) means:
        # u - v - x - y - z - w - u where all edges are color c.
        # So we need a path of length 4 from v to u in color c (not using edge (u,v)).
        # I.e., v - x - y - z - w - u, all color c.
        # This is equivalent to: exists x in N_c(v), w in N_c(u), such that
        # there exists y, z with y in N_c(x) ∩ N_c(z) and z in N_c(y) ∩ N_c(w)... 
        # This is getting complex.
        
        # Simpler approach: after adding edge (u,v) color c, check if there's a C6.
        # A C6 in color c is a cycle of 6 vertices. We can check by BFS/DFS.
        # But that's expensive.
        
        # Better: precompute all 6-cycles in K_n and check if any is monochromatic.
        # Number of 6-cycles in K_17 is C(17,6) * 5! / 2 = 12376 * 60 = 742560.
        # That's manageable!
        
        # Precompute all 6-cycles
        cycles_6 = []
        for verts in combinations(range(n), 6):
            # Number of distinct 6-cycles on 6 labeled vertices: (6-1)!/2 = 60
            # But we can just check all permutations that form a cycle
            # Actually, for a set of 6 vertices, the number of distinct cycles is 5!/2 = 60
            # Let's generate them
            v0, v1, v2, v3, v4, v5 = verts
            # Fix v0, then arrange v1..v5 in a circle: (5-1)!/2 = 12 distinct cycles
            # Wait, no. The number of distinct Hamiltonian cycles on 6 vertices is (6-1)!/2 = 60.
            # Let's just enumerate all 60.
            import itertools
            rest = [v1, v2, v3, v4, v5]
            for perm in itertools.permutations(rest):
                # Cycle: v0, perm[0], perm[1], perm[2], perm[3], perm[4], v0
                # To avoid duplicates, we can fix perm[0] < perm[4] or something
                # Actually, let's just add all and deduplicate later, or accept some redundancy
                cycle = (v0, perm[0], perm[1], perm[2], perm[3], perm[4])
                cycles_6.append(cycle)
        
        # This is 12376 * 120 = too many. Let's be smarter.
        # For each set of 6 vertices, there are 60 distinct cycles.
        # Total: C(17,6) * 60 = 12376 * 60 = 742560 cycles.
        # That's a lot but maybe manageable with bitsets.
        
        # Alternative approach: use the fact that we're doing backtracking.
        # At each step, we add one edge. We need to check if this creates a C6.
        # A C6 is created iff the new edge (u,v) color c completes a path of length 5
        # from u to v in color c, OR creates a shorter structure... no, only C6 matters.
        
        # Actually, adding edge (u,v) creates a C6 iff there exists a path of length 5
        # from u to v in the graph formed by color c edges (excluding the new edge).
        # Path of length 5: u - a - b - d - e - v, all color c.
        # This is equivalent to: exists a in N_c(u), e in N_c(v), such that
        # there's a path of length 3 from a to e in color c.
        # Path of length 3 from a to e: a - b - d - e, so exists b in N_c(a), d in N_c(e),
        # with b-d edge color c.
        
        # So: for a in N_c(u), for e in N_c(v), for b in N_c(a), for d in N_c(e):
        #     if (b,d) is an edge in color c, then C6 exists.
        
        # This is O(deg(u) * deg(v) * deg(a) * deg(e)) which is bad.
        
        # Let's precompute for each color c and each pair (x,y) whether there's a path
        # of length 3 from x to y in color c.
        # path3[c][x][y] = True if exists a,b with x-a, a-b, b-y all color c.
        
        # When we add edge (u,v) color c, a C6 is formed iff:
        # exists a in N_c(u), e in N_c(v) such that path3[c][a][e] is True.
        
        # We can maintain path3 incrementally.
        # Initially all False.
        # When we add edge (x,y) color c:
        #   For each a in N_c(x), for each e in N_c(y): path3[c][a][e] = True
        #   For each a in N_c(y), for each e in N_c(x): path3[c][a][e] = True
        #   Also, path3[c][x][y] might become True if there's a path of length 3,
        #   but that requires existing edges.
        
        # Actually, path3[c][a][e] becomes True when we add an edge that completes
        # a length-3 path. A length-3 path a-b-d-e requires edges (a,b), (b,d), (d,e).
        # When we add edge (b,d), if (a,b) and (d,e) already exist, then path3[c][a][e] = True.
        
        # This is getting complicated. Let's use a different approach.
        
        # Given the time constraints, let's try a randomized local search approach.
        # Start with a random 4-coloring of K_n, then iteratively fix violations.
        
        def random_coloring(n):
            num_edges = n * (n - 1) // 2
            return [random.randint(0, 3) for _ in range(num_edges)]
        
        def edge_index(n, i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        def has_mono_c6(n, colors):
            """Check if there's a monochromatic C6."""
            # Build adjacency per color
            adj = [set() for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_index(n, i, j)]
                    adj[c].add((i, j))
            
            # For each color, check for C6
            for c in range(4):
                # Build adjacency list
                adj_list = [[] for _ in range(n)]
                for (u, v) in adj[c]:
                    adj_list[u].append(v)
                    adj_list[v].append(u)
                
                # Check for C6 using BFS from each edge
                # A C6 containing edge (u,v) means there's a path of length 5 from u to v
                # without using edge (u,v)
                for (u, v) in adj[c]:
                    # BFS from u, avoiding edge (u,v), to see if we can reach v in 5 steps
                    # Actually, we need a path of length exactly 5
                    # Use BFS with depth tracking
                    visited = [[False] * 6 for _ in range(n)]  # visited[v][d] = reached v at depth d
                    queue = [(u, 0)]
                    visited[u][0] = True
                    found = False
                    while queue:
                        node, depth = queue.pop(0)
                        if depth == 5:
                            if node == v:
                                found = True
                                break
                            continue
                        for neighbor in adj_list[node]:
                            if neighbor == u and depth == 0:
                                continue  # avoid going back on edge (u,v)
                            if neighbor == v and depth < 5:
                                if depth + 1 == 5:
                                    found = True
                                    break
                                continue  # can't use v before depth 5
                            if not visited[neighbor][depth + 1]:
                                visited[neighbor][depth + 1] = True
                                queue.append((neighbor, depth + 1))
                        if found:
                            break
                    if found:
                        return True
            return False
        
        # Try n = 17 first, then 16, etc.
        for n in [17, 16, 15]:
            if time.time() > deadline - 5:
                break
            
            # Try multiple random restarts
            best_result = None
            num_attempts = 0
            while time.time() < deadline - 10:
                num_attempts += 1
                colors = random_coloring(n)
                
                # Local search: try to fix violations
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    if not has_mono_c6(n, colors):
                        best_result = colors[:]
                        break
                    # Find a violating C6 and change one of its edges
                    # For simplicity, just randomly change some edges
                    for _ in range(100):
                        idx = random.randint(0, len(colors) - 1)
                        colors[idx] = random.randint(0, 3)
                        if not has_mono_c6(n, colors):
                            best_result = colors[:]
                            improved = True
                            break
                
                if best_result is not None:
                    break
            
            if best_result is not None:
                return {'n': n, 'colors': best_result}
        
        # If we couldn't find a good one, return a small valid coloring
        # K9 with circular coloring
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    return find_coloring(17, deadline)

# EVOLVE-BLOCK-END
