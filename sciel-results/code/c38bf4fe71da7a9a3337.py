"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    base_edges = P['edges']
    
    # Build base adjacency
    base_adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        base_adj[u][v] = True
        base_adj[v][u] = True
    
    # Precompute all candidate edges (non-edges that don't create triangles with base)
    # In the base path, adding edge (u,v) creates a triangle iff there exists w adjacent to both u and v
    # In a path, u is adjacent to u-1 and u+1 (if they exist). So (u,v) creates a triangle iff
    # v is adjacent to u-1 or u+1, i.e., v = u-2 or v = u+2 (since v can't be u-1 or u+1 as those are already edges)
    # Wait, let me think more carefully. In the path 0-1-2-...-23:
    # Neighbors of u: {u-1, u+1} (excluding endpoints)
    # Neighbors of v: {v-1, v+1}
    # Triangle (u,v,w) exists iff w is neighbor of both u and v.
    # w is neighbor of u means w in {u-1, u+1}
    # w is neighbor of v means w in {v-1, v+1}
    # So we need {u-1, u+1} ∩ {v-1, v+1} ≠ ∅
    # This means u-1 = v-1 (u=v, impossible), u-1 = v+1 (v = u-2), u+1 = v-1 (v = u-2), u+1 = v+1 (u=v, impossible)
    # So triangle is created iff |u - v| = 2
    
    # So valid single edges are all (u,v) with |u-v| >= 3 (and not already in base, which means |u-v| >= 1)
    # Base edges have |u-v| = 1, so valid edges have |u-v| >= 3
    
    candidate_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if v - u >= 3:  # |u-v| >= 3 means no triangle with base path
                candidate_edges.append((u, v))
    
    # BFS diameter computation on a given adjacency matrix
    def bfs_diameter(adj):
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = du + 1
                        queue.append(v)
            for d in dist:
                if d == -1:
                    return -1  # disconnected
                if d > max_diam:
                    max_diam = d
        return max_diam
    
    # Check if adding a set of edges to base keeps it triangle-free
    # An edge (u,v) is compatible with current adjacency if no common neighbor
    def is_triangle_free_with_current(adj, edge_set):
        """Check all pairs in edge_set don't create triangles with each other or with base."""
        for (u, v) in edge_set:
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    return False
        return True
    
    # Full validation: triangle-free, connected, diameter <= 4
    def validate(added_edges):
        adj = [row[:] for row in base_adj]
        for u, v in added_edges:
            # Check no triangle
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    return False
            adj[u][v] = True
            adj[v][u] = True
        return bfs_diameter(adj) <= 4
    
    # Strategy: Systematic search for minimal edge sets
    # Key insight: We need diameter <= 4. The path has diameter 23.
    # With k added edges, we can significantly reduce distances.
    
    # Let's use a smart approach:
    # 1. First, find a good initial solution
    # 2. Then try to reduce it
    
    # Initial solution: Use vertex 12 as a hub, connect to strategic vertices
    # But we need to be smarter. Let's think about what's needed for diameter 4.
    
    # For diameter 4, every pair of vertices must be within 4 hops.
    # The path vertices 0 and 23 are 23 apart. We need shortcuts.
    
    # Let's try a structured approach:
    # Consider that with edges from a hub to several vertices, we can create shortcuts.
    # But triangles are forbidden, so a hub can only connect to vertices that are 
    # not adjacent to each other's neighbors.
    
    # For hub h, vertex v can connect to h iff |h-v| >= 3 (no triangle with path)
    # AND for any other vertex u also connected to h, we need: 
    #   no common neighbor of u and v in the augmented graph.
    # In the base path, common neighbors of u and v exist iff |u-v| = 2.
    # After adding edges, new common neighbors can appear.
    
    # Let's try a direct search approach with pruning:
    
    # Precompute: for each pair of candidate edges, check if they're mutually compatible
    # (don't create triangles with each other)
    
    # Two edges (a,b) and (c,d) are incompatible if they share a vertex and the other endpoints 
    # have a common neighbor, OR if they form a triangle together.
    # Actually, the simplest check: after adding both edges, is the graph triangle-free?
    # Triangle (x,y,z) exists if all three pairs (x,y), (y,z), (z,x) are edges.
    # New triangles can only involve at least one new edge.
    
    # For efficiency, let's precompute compatibility between candidate edges.
    # Two candidate edges e1=(a,b) and e2=(c,d) are incompatible if adding both creates a triangle.
    # A triangle requires 3 vertices all pairwise connected.
    # Cases:
    # 1. They share a vertex: e1=(a,b), e2=(a,c). Triangle if b and c have a common neighbor.
    #    In base: common neighbor iff |b-c|=2. After adding: also check if b-c is an edge (it's not in base since |b-c|>=3 for candidates... wait no, b and c could be close).
    #    Actually b and c could be anything. Let me just check: is there w adjacent to both b and c in base_adj?
    #    Or is b=c? No, edges are distinct.
    #    Or is (b,c) itself an edge in the augmented graph? If |b-c|>=3 it's not in base. If |b-c|=1 it's in base. If |b-c|=2 it's not in base but creates triangles.
    #    Hmm, this is getting complex. Let me just do the direct check for small sets.
    
    # Given the small size (24 vertices, ~250 candidate edges), let's use a different strategy:
    # Greedy set cover approach for diameter reduction.
    
    # For each pair of vertices (i,j) with base distance > 4, we need a "shortcut" path of length <= 4.
    # A shortcut can use 0, 1, 2, or 3 added edges.
    
    # Actually, let me just do a focused search:
    # Try all solutions of size 5, 6, 7 using smart enumeration.
    
    # For size k, we need to pick k edges from ~250 candidates. C(250,5) is too large.
    # But we can use constraints to prune.
    
    # Better approach: Use the structure of the path.
    # Key observation: For diameter 4, the "middle" of the path needs to be well-connected.
    # Vertices near the ends (0,1,2 and 21,22,23) are hardest to reach.
    
    # Let me try a different structural approach:
    # Think of it as: we need to add edges so that the graph has a "core" that everything can reach within 4 steps.
    
    # Let's try specific patterns:
    # Pattern 1: Star-like from one or two hubs
    # Pattern 2: Chain of shortcuts
    # Pattern 3: Mixed
    
    # Let me try to find a 5-edge or 6-edge solution by construction.
    
    # For diameter 4, consider that if we add edge (a,b) with |a-b| >= 3, it creates a shortcut.
    # The distance from x to y in the augmented graph can be shortened if there's a path using added edges.
    
    # Let me try: connect vertices to create a "ladder" structure.
    # E.g., edges: (0,4), (4,8), (8,12), (12,16), (16,20), (20,23) - but need to check triangles.
    # (0,4): |0-4|=4, ok. (4,8): |4-8|=4, ok. But do (0,4) and (4,8) create a triangle?
    # They share vertex 4. Need common neighbor of 0 and 8. In base, 0's neighbors: {1}. 8's neighbors: {7,9}. No common. OK.
    # But wait, after adding (0,4), 0's neighbors include 4. After adding (4,8), 8's neighbors include 4.
    # Triangle (0,4,8)? Need 0-8 edge. |0-8|=8, not in base, not added. OK.
    # Triangle (0,4,7)? Need 0-7 and 4-7. 4-7: |4-7|=3, not in base. 0-7: no. OK.
    # Actually I need to be more careful. Let me just validate programmatically.
    
    # Let me try a few hand-crafted solutions and validate:
    
    test_solutions = [
        # 6-edge attempt
        [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23)],
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23), (0, 3)],
        [(1, 5), (5, 9), (9, 13), (13, 17), (17, 21), (21, 23)],
        # 5-edge attempts
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)],
        [(0, 6), (6, 12), (12, 18), (18, 23), (0, 3)],
        [(1, 6), (6, 11), (11, 16), (16, 21), (21, 23)],
        [(0, 4), (4, 9), (9, 14), (14, 19), (19, 23)],
        [(0, 5), (5, 11), (11, 17), (17, 22), (0, 23)],
    ]
    
    best = None
    best_count = 999
    
    for sol in test_solutions:
        if time.time() > deadline:
            break
        if validate(sol):
            if len(sol) < best_count:
                best = sol
                best_count = len(sol)
    
    # If no small solution found, use a known-good larger solution
    if best is None:
        # Use the seed solution
        seed = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
        if validate(seed):
            best = seed
            best_count = len(seed)
        else:
            # Try another fallback
            fallback = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23), (0, 2), (22, 23)]
            # Wait, (0,2) creates triangle with (0,1) and (1,2). Bad.
            fallback = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23), (1, 5), (22, 23)]
            # (22,23) is already a base edge! Bad.
            fallback = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23), (1, 5), (21, 23)]
            if validate(fallback):
                best = fallback
                best_count = len(fallback)
            else:
                # Last resort: very conservative
                best = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23), (1, 4), (21, 23), (2, 6)]
                if not validate(best):
                    best = [(0, 6), (6, 12), (12, 18), (18, 23), (1, 4), (4, 8), (14, 17), (20, 23)]
                    if not validate(best):
                        # Absolute last resort
                        best = [(0, 4), (4, 9), (9, 14), (14, 19), (19, 23), (1, 6), (6, 11), (11, 16), (16, 21), (21, 23)]
    
    # Now try to improve: local search to reduce edge count
    if time.time() < deadline - 5:
        # Try removing one edge and see if still valid
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline - 5:
                    break
                candidate = best[:i] + best[i+1:]
                if validate(candidate):
                    best = candidate
                    improved = True
                    break
    
        # Try swapping: remove one, add one different
        if time.time() < deadline - 10:
            for i in range(len(best)):
                if time.time() > deadline - 10:
                    break
                current = best[:i] + best[i+1:]
                current_set = set(current)
                for cand in candidate_edges:
                    if time.time() > deadline - 10:
                        break
                    if cand in current_set:
                        continue
                    new_sol = current + [cand]
                    if len(new_sol) < len(best) and validate(new_sol):
                        best = new_sol
                        break
    
    # Final validation
    if not validate(best):
        # This should not happen, but just in case
        best = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    
    return [list(e) for e in best]

# EVOLVE-BLOCK-END
