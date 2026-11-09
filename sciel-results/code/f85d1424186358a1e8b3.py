import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Target n values to try, descending
    target_ns = [17, 16, 15, 14, 13, 12, 11, 10, 9]
    
    # Precompute all 6-combinations of vertices for n=17?
    # It's better to generate on the fly or precompute for max n=17 and filter.
    # But n changes.
    
    def count_c6(coloring, n):
        """
        coloring: list of colors for edges (i,j) in lexicographic order.
        Returns total number of monochromatic C6 cycles.
        """
        # Build adjacency matrices for each color
        # 4 colors, n x n matrices
        # Using numpy for speed?
        # Or just sets of edges.
        
        # Let's use sets of tuples (u, v) with u < v
        edges_by_color = [set() for _ in range(4)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = coloring[idx]
                edges_by_color[c].add((i, j))
                idx += 1
        
        total_cycles = 0
        
        # Iterate over all combinations of 6 vertices
        # For each combination, check if it forms a C6 in any color
        # A set of 6 vertices {v0..v5} forms a C6 if the edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0) are all in the same color.
        # There are 6! / 12 = 60 distinct cycles on 6 labeled vertices?
        # Actually, the number of Hamiltonian cycles in K6 is (6-1)!/2 = 60.
        # So for each set of 6 vertices, there are 60 possible cycles.
        # We need to check if ANY of these 60 cycles is monochromatic.
        # If multiple are, count them? The objective is to minimize the count.
        
        # Optimization:
        # For a specific set of 6 vertices, we can check all 60 permutations.
        # This might be slow in pure Python.
        
        # Alternative:
        # For each color c, count the number of C6 in the subgraph induced by that color.
        # Sum over c.
        
        # Let's try to optimize the cycle count.
        # For a fixed color c, we have a graph G_c.
        # We want to count 6-cycles in G_c.
        # We can iterate over all 6-sets. For a 6-set S, if the induced subgraph G_c[S] contains a C6, count it.
        # How many C6 can a 6-set contain in a simple graph?
        # The complete graph K6 contains 60 C6s.
        # A sparse graph contains fewer.
        
        # Given n=17, total 6-sets is 12376.
        # For each 6-set, checking 60 cycles is 12376 * 60 * 6 edge checks = ~4.4 million checks.
        # This is doable in ~1-2 seconds in Python if optimized.
        # We need to do this many times.
        
        # Better: Use the adjacency matrix and matrix multiplication?
        # No, let's stick to the set-based approach but optimize the inner loop.
        
        # Actually, we can just check if the induced subgraph is "C6-free".
        # But we need a score for SA.
        
        # Let's use a simpler heuristic:
        # Score = number of "bad" edges? No.
        # Score = number of monochromatic C6.
        
        # To speed up:
        # Precompute all 60 cycle patterns (permutations of 0..5 that form a cycle).
        # For a 6-set S = [v0, v1, v2, v3, v4, v5], map to 0..5.
        # For each pattern P, check if edges (S[P[i]], S[P[i+1]]) are in color c.
        
        # This is still heavy.
        
        # Alternative Strategy:
        # Instead of counting ALL cycles, just find IF any exist.
        # If we use a "repair" strategy:
        # 1. Random coloring.
        # 2. Find one monochromatic C6.
        # 3. Break it by recoloring one edge.
        # 4. Repeat.
        # This requires finding a C6 fast.
        
        # Finding a C6:
        # Iterate 6-sets. For each, check if it has a monochromatic C6.
        # Stop at the first one found.
        
        pass

    # Implementation of the repair strategy
    def find_c6(coloring, n):
        """Returns a tuple of 6 vertices forming a monochromatic C6, or None."""
        # Build edges_by_color
        edges_by_color = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = coloring[idx]
                edges_by_color[c].add((i, j))
                idx += 1
        
        # Precomputed cycle patterns on 6 nodes (indices 0-5)
        # We need all permutations of 0..5 that form a cycle.
        # A cycle is a permutation p such that p[0]->p[1]->...->p[5]->p[0] are edges.
        # We can generate these once.
        
        # Actually, we can just check the 60 unique cycles.
        # To avoid regenerating, we can hardcode or generate at start.
        
        # Let's generate all 60 cycles for the canonical 6-set [0,1,2,3,4,5]
        # and then map them to the actual vertices.
        
        # There are 60 distinct cycles in K6.
        # We can represent them as tuples of 6 vertices (relative indices).
        # e.g., (0,1,2,3,4,5) is a cycle.
        
        # Let's just iterate over all 6-sets of vertices.
        # For each 6-set, we check if there is a monochromatic cycle.
        
        # To speed up, we can check each color separately.
        # For color c, check if the induced subgraph on the 6-set has a C6.
        
        # This is still potentially slow if we do it naively.
        
        # Let's try a different approach:
        # Use a SAT solver? No, stdlib only.
        
        # Let's use the "Count C6" with NumPy for the whole graph?
        # No, the graph changes.
        
        # Let's try to optimize the "Find C6" loop.
        # 1. Build adjacency list for each color.
        # 2. For each 6-set, check if it contains a C6.
        
        # We can prune: If a 6-set has a vertex with degree < 2 in ALL colors? No, degree in a specific color.
        # If for a 6-set S, and for every color c, the induced subgraph G_c[S] is not 2-connected, it can't have a C6.
        # A C6 requires min degree 2 in the induced subgraph for that color.
        
        # Let's implement the basic version first and see if it's fast enough.
        
        # Precompute all 60 cycles of K6
        # A cycle is a permutation of 0..5 where we identify rotations and reflections.
        # We can just list them.
        
        # Actually, we can just pick a random 6-set and check.
        
        # Let's return None for now and implement the logic in the main loop.
        return None

    # This is getting complex. Let's use a simpler, known heuristic for Ramsey colorings.
    # "Random greedy with backtracking" or "Simulated Annealing on edge colors".
    
    # Given the time limit, I will implement a robust SA that minimizes the number of monochromatic C6s.
    # I will use a fast C6 counter based on bitsets or sets.
    
    # For n=17, edges=136.
    # We will maintain the count of C6s for each color.
    # Updating the count when an edge changes is the key.
    
    # Let's try a simpler path:
    # 1. Start with n=17.
    # 2. Random coloring.
    # 3. Loop:
    #    a. Find a monochromatic C6. (If none, done).
    #    b. Recolor one edge of that C6 to a different color.
    #    c. If stuck (no progress in X steps), restart with random coloring.
    
    # Finding a monochromatic C6 efficiently:
    # Iterate over all 6-sets.
    # For each 6-set, check if any color forms a C6.
    
    # Let's code this.
    
    def get_edges_by_color(coloring, n):
        ebc = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = coloring[idx]
                ebc[c].add((i, j))
                idx += 1
        return ebc

    def has_c6_in_set(vertices, ebc):
        """
        vertices: tuple of 6 ints
        ebc: list of sets of edges
        Returns color if a C6 exists in that color, else -1
        """
        v = vertices
        # Check each color
        for c in range(4):
            # Check if there is a cycle in ebc[c]
            # We need to check the 60 permutations.
            # To avoid generating 60 permutations every time, we can precompute them.
            # But the vertices are different.
            # We can map v to 0..5.
            # Let's just check the specific edges of the "standard" cycles.
            
            # Actually, it's faster to check if the induced subgraph is a C6 or contains one.
            # For a 6-set, the only way to have a C6 is if the induced subgraph contains a cycle of length 6.
            # Since the set has exactly 6 vertices, a C6 must use all of them.
            # So we just need to check if the induced subgraph on these 6 vertices contains a Hamiltonian cycle.
            
            # We can check this by trying all 60 permutations.
            # To speed up, we can precompute the 60 permutations of (0,1,2,3,4,5) that form a cycle.
            # Let's do that globally.
            
            pass
        return -1

    # Precompute cycle patterns
    # A pattern is a tuple of 6 distinct indices from 0..5.
    # We consider (0,1,2,3,4,5) and its rotations/reflections equivalent?
    # No, we just need to check if edges (p[0],p[1]), (p[1],p[2]), (p[2],p[3]), (p[3],p[4]), (p[4],p[5]), (p[5],p[0]) are in the set.
    
    # There are 60 such patterns.
    # Let's generate them.
    cycle_patterns = []
    # Fix 0 at the start to reduce symmetry?
    # No, just generate all permutations of 1..5 and check if they form a cycle with 0.
    # Actually, just generate all permutations of 0..5 and filter those that are cycles.
    # But we can just hardcode the check for a specific set of edges.
    
    # Let's use a different metric:
    # For each 6-set, count the number of edges in each color.
    # If a color has 6 edges, it might be a C6.
    # If a color has < 6 edges, it can't be a C6 (since C6 has 6 edges).
    # So, for a 6-set, if no color has exactly 6 edges (i.e., is a complete graph K6? No, C6 is not K6),
    # wait. A C6 has 6 edges. The induced subgraph on 6 vertices for a C6 has exactly 6 edges.
    # So, if for a 6-set S, the number of edges of color c in S is not 6, it cannot be a C6 of color c.
    # If it is 6, it could be a C6, or it could be a different graph with 6 edges (e.g., K4 + 2 isolated? No, 6 vertices).
    # Graphs with 6 vertices and 6 edges: C6, K3+C3, etc.
    # So we only need to check the 6-sets where some color has exactly 6 edges.
    # This is a huge pruning!
    
    # Let's implement this.
    
    def find_c6(coloring, n):
        ebc = get_edges_by_color(coloring, n)
        
        # Iterate over all 6-sets
        for verts in itertools.combinations(range(n), 6):
            # Count edges per color in this 6-set
            counts = [0, 0, 0, 0]
            for i in range(6):
                for j in range(i+1, 6):
                    u = verts[i]
                    v = verts[j]
                    # Find color of (u,v)
                    # We need a fast lookup.
                    # ebc is sets. We can check membership.
                    # But we need to know which color.
                    # Let's store edge -> color in a dict or array.
                    pass
        
        # This requires a fast edge color lookup.
        # Let's create a 2D array or dict.
        
        pass

    # Refined Plan:
    # 1. Create a `color_matrix` of size n x n.
    # 2. Iterate 6-sets.
    # 3. For each 6-set, count edges per color.
    # 4. If any count is 6, check if the 6 edges form a C6.
    # 5. If yes, return the 6-set and the color.
    
    # Checking if 6 edges form a C6:
    # The 6 edges must form a single cycle of length 6.
    # We can check the degrees. In a C6, all 6 vertices have degree 2.
    # So, if the induced subgraph has 6 edges and all vertices have degree 2, it is a C6 (or two C3s? No, C3+C3 has 6 edges and degrees 2,2,2,2,2,2).
    # Wait, C3+C3 is two disjoint triangles.
    # So we need to distinguish C6 from C3+C3.
    # A C6 is connected. C3+C3 is not.
    # So: If 6 edges, all degrees 2, and connected -> C6.
    
    # This is fast!
    
    def is_c6(verts, color_matrix):
        # verts is a tuple of 6 vertices
        # Check if the induced subgraph (using color_matrix? No, we need the color)
        # This function should be called only if we know the color.
        pass

    # Let's integrate this into the main loop.
    
    def solve_n(n):
        # Initialize random coloring
        num_edges = n * (n - 1) // 2
        coloring = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Build lookup
        # We will maintain the coloring list and rebuild ebc/lookup when needed.
        
        best_coloring = coloring
        best_score = float('inf')
        
        # We will use a simple hill climbing / repair.
        # Score = number of monochromatic C6s.
        # But counting them all is slow.
        # Let's just find ONE C6. If found, repair.
        
        for step in range(10000):
            if time.time() > deadline:
                break
            
            # Find a C6
            c6_info = find_any_c6(coloring, n)
            
            if c6_info is None:
                return coloring # Success!
            
            # c6_info: (verts, color)
            verts, c = c6_info
            
            # Pick a random edge from the C6
            # The edges of the C6 are the ones of color c in this 6-set.
            # We need to identify them.
            # Since we know it's a C6, there are exactly 6 edges of color c.
            # Let's find them.
            edges_in_c6 = []
            for i in range(6):
                for j in range(i+1, 6):
                    u = verts[i]
                    v = verts[j]
                    # Check if edge (u,v) is color c
                    # We need a fast lookup.
                    pass
            
            # Recolor one edge to a random different color
            # Update coloring
            
            pass
        
        return None # Failed

    # This looks promising. The bottleneck is `find_any_c6`.
    # With the "count edges per color" pruning, it should be fast.
    
    # Let's write the code.

# Final Code Structure
import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Try n=17 first
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 5:
            break
            
        result = try_solve(n, deadline)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def try_solve(n, deadline):
    num_edges = n * (n - 1) // 2
    
    # Helper to get edge index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + j - i - 1
        # Formula: sum_{k=0}^{i-1} (n-k-1) + (j-i-1)
        # Simpler: precompute a dict or use a 2D array.
        # For n=17, 2D array is fine.
        
    # Let's use a 2D array for fast access
    # colors[i][j] = c
    
    def make_matrix(coloring):
        mat = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = coloring[idx]
                mat[i][j] = c
                mat[j][i] = c
                idx += 1
        return mat

    def find_c6(mat):
        # Iterate 6-sets
        for verts in itertools.combinations(range(n), 6):
            counts = [0, 0, 0, 0]
            # Count edges per color
            for i in range(6):
                for j in range(i+1, 6):
                    c = mat[verts[i]][verts[j]]
                    counts[c] += 1
            
            # Check if any color has 6 edges
            for c in range(4):
                if counts[c] == 6:
                    # Check if it forms a C6
                    # Degrees must be 2 for all vertices
                    # And it must be connected (or just not C3+C3)
                    
                    # Build subgraph edges for this color
                    # Check degrees
                    deg = [0]*6
                    for i in range(6):
                        for j in range(i+1, 6):
                            if mat[verts[i]][verts[j]] == c:
                                deg[i] += 1
                                deg[j] += 1
                                
                    if all(d == 2 for d in deg):
                        # Could be C6 or C3+C3
                        # Check connectivity or just check if it's a single cycle
                        # Simple check: try to traverse
                        # Start at 0, follow edges
                        visited = [False]*6
                        visited[0] = True
                        cur = 0
                        for _ in range(5):
                            # Find next
                            found = False
                            for k in range(6):
                                if not visited[k] and mat[verts[cur]][verts[k]] == c:
                                    visited[k] = True
                                    cur = k
                                    found = True
                                    break
                            if not found:
                                break
                        if all(visited):
                            # It's a single cycle
                            return verts, c
                            
        return None

    # Main loop
    best_coloring = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        coloring = [random.randint(0, 3) for _ in range(num_edges)]
        mat = make_matrix(coloring)
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            c6_info = find_c6(mat)
            
            if c6_info is None:
                return coloring # Success
            
            verts, c = c6_info
            
            # Find the 6 edges of color c in this 6-set
            edges = []
            for i in range(6):
                for j in range(i+1, 6):
                    if mat[verts[i]][verts[j]] == c:
                        edges.append((verts[i], verts[j]))
                        
            # Pick a random edge
            u, v = random.choice(edges)
            
            # Recolor to a different color
            new_c = random.choice([0, 1, 2, 3])
            while new_c == c:
                new_c = random.choice([0, 1, 2, 3])
                
            # Update coloring and mat
            idx = edge_idx(u, v)
            coloring[idx] = new_c
            mat[u][v] = new_c
            mat[v][u] = new_c
            
        # If we didn't find a solution in this restart, we could try to keep the best one?
        # But "best" is hard to define without a score.
        # Let's just restart.
        
    return None

# Need to implement edge_idx correctly
def edge_idx(i, j, n):
    if i > j:
        i, j = j, i
    # Number of edges before i: sum_{k=0}^{i-1} (n - k - 1)
    # = i*n - i - i*(i-1)/2
    # Edges in row i: j - i - 1
    return i * n - i - i * (i - 1) // 2 + j - i - 1

# Wait, the formula `i * n - i - i * (i - 1) // 2` is for 0-indexed rows?
# Row 0: n-1 edges.
# Row 1: n-2 edges.
# ...
# Row i: n-i-1 edges.
# Total edges before row i: sum_{k=0}^{i-1} (n-k-1) = i(n-1) - i(i-1)/2
# Index in row i: j - i - 1
# Total: i(n-1) - i(i-1)/2 + j - i - 1
# Let's test n=3.
# (0,1): i=0, j=1. 0 + 0 + 0 = 0. Correct.
# (0,2): i=0, j=2. 0 + 0 + 1 = 1. Correct.
# (1,2): i=1, j=2. 1(2) - 0 + 0 = 2. Correct.
# Formula: i * (n - 1) - i * (i - 1) // 2 + j - i - 1

# Let's refine the code.

