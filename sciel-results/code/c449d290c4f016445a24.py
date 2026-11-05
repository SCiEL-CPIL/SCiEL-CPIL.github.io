"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
import time
import random
import itertools
from collections import deque

# Deadline guard
_DEADLINE = 120.0  # seconds total allowed
_SAFETY_MARGIN = 5.0  # seconds to return before hard limit


def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj (n x n) contains a 5-cycle."""
    # For each pair of non-adjacent vertices (u, v), check if there's a path of length 3 between them
    # that doesn't use edge (u,v) directly, forming a C5.
    # Actually, a C5 exists iff there are two non-adjacent vertices u, v with a common neighbor w1 of u
    # and w2 of v such that... let me think more carefully.
    # 
    # A C5 is a cycle v0-v1-v2-v3-v4-v0. 
    # Check: for all pairs (i,j) with i<j, if not adj[i][j], check if there exists a path of length 4
    # from i to j that doesn't repeat vertices. But that's expensive.
    #
    # Better: For each unordered pair (u,v) that are NOT adjacent, check if there exist distinct 
    # vertices a, b, c such that u-a-b-c-v is a path (all edges exist, all vertices distinct, 
    # a != v, c != u, a != b, b != c).
    # This means: adj[u][a] and adj[a][b] and adj[b][c] and adj[c][v] with u,v,a,b,c all distinct
    # and u not adjacent to v (which we already check).
    #
    # To speed up: for each non-edge (u,v), look at neighbors of u (excluding v) and neighbors of v (excluding u).
    # We need a->b->c where a in N(u)\{v}, c in N(v)\{u}, a != c, and b is common neighbor of a and c.
    # Also need a != c and all 5 vertices distinct.
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                continue
            # u and v are not adjacent. Check for path u-a-b-c-v
            nu = [x for x in range(n) if adj[u][x] and x != v]
            nv = [x for x in range(n) if adj[v][x] and x != u]
            # For each a in nu, c in nv with a != c and a != v, c != u (already ensured),
            # check if there exists b adjacent to both a and c, with b not in {u,v,a,c}
            for a in nu:
                for c in nv:
                    if a == c:
                        continue
                    # Need b adjacent to a and c, b not in {u, v, a, c}
                    # Common neighbors of a and c
                    ca = adj[a]
                    cc = adj[c]
                    # b must be adjacent to a and c
                    for b in range(n):
                        if b == u or b == v or b == a or b == c:
                            continue
                        if ca[b] and cc[b]:
                            return True
    return False


def _has_independent_set_of_size_9(adj, n):
    """Check if there's an independent set of size 9 in the graph."""
    # Use branch and bound: pick a vertex, either include it (remove its neighbors) or exclude it.
    # We want to find if there's a set of 9 mutually non-adjacent vertices.
    
    best_found = [0]
    
    def branch(remaining, size):
        if size == 9:
            return True
        if len(remaining) < 9 - size:
            return False
        if size + len(remaining) < 9:
            return False
        # Pick a vertex
        # Heuristic: pick vertex with highest degree in the induced subgraph on remaining
        # (to prune more)
        best_v = -1
        best_deg = -1
        for v in remaining:
            deg = sum(adj[v][u] for u in remaining)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if best_v == -1:
            return False
        
        # Option 1: include best_v in independent set
        # Remove best_v and all its neighbors from remaining
        new_remaining = [u for u in remaining if u != best_v and not adj[best_v][u]]
        if branch(new_remaining, size + 1):
            return True
        
        # Option 2: exclude best_v
        new_remaining = [u for u in remaining if u != best_v]
        if branch(new_remaining, size):
            return True
        
        return False
    
    remaining = list(range(n))
    return branch(remaining, 0)


def _count_independent_sets_of_size_k(adj, n, k=9):
    """Count independent sets of size k (for verification)."""
    count = [0]
    
    def branch(remaining, size):
        if size == k:
            count[0] += 1
            return
        if len(remaining) < k - size:
            return
        if size + len(remaining) < k:
            return
        
        v = remaining[0]
        rest = remaining[1:]
        
        # Include v
        new_remaining = [u for u in rest if not adj[v][u]]
        branch(new_remaining, size + 1)
        
        # Exclude v
        branch(rest, size)
    
    branch(list(range(n)), 0)
    return count[0]


def construct():
    start_time = time.time()
    
    # Try to build a graph with n=35 first, then 34, 33, 32
    # Strategy: use a structured construction + random edge additions
    
    # Base: use a known good structure. 
    # C5-free graphs: bipartite graphs are C5-free. 
    # A bipartite graph with parts of size a and b has independence number max(a,b) if complete bipartite,
    # but for sparse bipartite graphs, the independence number can be larger.
    # 
    # Actually, for a bipartite graph with parts L and R, any subset of L is independent, so alpha >= |L|.
    # To have alpha < 9, we need |L| <= 8 and |R| <= 8, giving n <= 16. That's too small.
    #
    # So we need non-bipartite C5-free graphs. 
    # C5-free graphs can be: triangle-free (bipartite), or have triangles but no C5.
    # 
    # Let's try: start with a random graph and remove edges to eliminate C5s while keeping alpha < 9.
    # Or: start with a dense graph and sparsify.
    
    # Alternative approach: use a known construction.
    # The Ramsey number R(C5, K9): we need a graph with no C5 and no independent set of 9.
    # 
    # Let's try a random approach:
    # 1. Start with a random graph G(n, p) for various p
    # 2. Find all C5s and remove one edge from each
    # 3. Check if alpha < 9
    # 4. If alpha >= 9, try to add edges to reduce alpha (while maintaining C5-free)
    
    # Actually, let's try a different structural approach:
    # Use a blow-up of a small C5-free graph.
    # 
    # For example, take K4 (which is C5-free) and blow up each vertex to a set of size s.
    # Between two blown-up parts, include all edges (complete bipartite).
    # This gives a graph with 4s vertices.
    # C5-free? A C5 in the blow-up would correspond to a C5 in K4, which doesn't exist.
    # Wait, but within a part, there are no edges. So a C5 would need to visit 5 different parts,
    # but K4 has only 4 vertices, so no C5. Yes, C5-free.
    # Independence number: in the blow-up, we can pick at most one vertex from each part (since parts
    # are complete to each other... wait, no. Within a part, no edges, so we can pick ALL vertices from
    # one part. And between parts, all edges exist, so we can only pick from one part.
    # So alpha = s (pick all of one part).
    # To have alpha < 9, need s <= 8. So n = 4*8 = 32.
    # 
    # Let's verify: 4 parts of size 8, complete between parts, no edges within parts.
    # n = 32, alpha = 8 < 9. C5-free because any cycle must use at least 2 vertices from some part
    # (pigeonhole, 5 vertices, 4 parts), but no edges within a part, so no cycle can use 2 from same part.
    # Wait, that's not right. A cycle v0-v1-v2-v3-v4-v0. If two consecutive vertices are in the same part,
    # there's no edge. So consecutive vertices must be in different parts. With 5 vertices and 4 parts,
    # by pigeonhole, two vertices are in the same part. But they might not be consecutive.
    # Example: v0 in A, v1 in B, v2 in A, v3 in C, v4 in D. Then v0-v2 are both in A but not adjacent in cycle.
    # Edges: v0(A)-v1(B) ok, v1(B)-v2(A) ok, v2(A)-v3(C) ok, v3(C)-v4(D) ok, v4(D)-v0(A) ok.
    # This is a valid C5! So the blow-up of K4 is NOT C5-free.
    
    # So that construction doesn't work. Let me think differently.
    
    # C5-free means no 5-cycle. Graphs with no C5 include:
    # - Forests
    # - Complete graphs K_n for n < 5 (K4 has no C5)
    # - Complete bipartite graphs
    # - More generally, graphs of treewidth <= 2 are C5-free? No, a C5 has treewidth 2.
    # 
    # Let's try: start with n=35, random edges, then iteratively remove edges to break C5s.
    # Then check alpha. If alpha >= 9, we're stuck (can't add edges without risking C5).
    # 
    # Better: use a SAT-like or local search approach.
    # 
    # For now, let's try the 4-part construction with n=32 and verify it works,
    # then try to extend to 33, 34, 35.
    
    # Actually, let me reconsider. The task says "best known has 32 vertices".
    # So n=32 might be achievable. Let me try to construct it.
    
    # Try: complete 4-partite graph with parts of sizes 8,8,8,8.
    # n=32. As shown above, this has a C5. So it doesn't work.
    
    # Try: complete 3-partite graph with parts of sizes 11,11,11? n=33.
    # C5 in complete 3-partite: v0(A),v1(B),v2(A),v3(C),v4(B). 
    # Edges: A-B ok, B-A ok, A-C ok, C-B ok, B-A ok. Valid C5. So no.
    
    # Try: complete 2-partite (bipartite) with parts 17,18? n=35.
    # alpha = max(17,18) = 18 >= 9. No good.
    
    # We need a non-complete multipartite graph.
    
    # Let's try: take K4 and subdivide each edge? No, that creates C6s and possibly C5s.
    
    # Alternative: use a random graph with controlled density and repair.
    
    # Let's implement a local search:
    # 1. Start with random graph G(35, p) for p around 0.3-0.5
    # 2. While there's a C5, remove a random edge from a C5
    # 3. Check alpha. If alpha < 9, done.
    # 4. If alpha >= 9, try to add edges between independent vertices (carefully)
    
    # But this might not work well. Let me try a different approach.
    
    # Known result: R(C5, K9) > 32 means there exists a 32-vertex C5-free graph with alpha <= 8.
    # Let's try to find one via random search.
    
    best_n = 28
    best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]
    
    for n_target in [35, 34, 33, 32]:
        if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Try multiple random instances
        for trial in range(50):
            if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
                break
            
            random.seed(trial * 1000 + n_target)
            
            # Start with a sparse random graph
            p = 0.3
            adj = [[False] * n_target for _ in range(n_target)]
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < p:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Remove C5s
            max_iter = 1000
            for it in range(max_iter):
                if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                # Find a C5
                found = False
                for u in range(n_target):
                    for v in range(u + 1, n_target):
                        if adj[u][v]:
                            continue
                        nu = [x for x in range(n_target) if adj[u][x] and x != v]
                        nv = [x for x in range(n_target) if adj[v][x] and x != u]
                        for a in nu:
                            for c in nv:
                                if a == c:
                                    continue
                                for b in range(n_target):
                                    if b in (u, v, a, c):
                                        continue
                                    if adj[a][b] and adj[b][c]:
                                        # Found C5: u-a-b-c-v
                                        # Remove a random edge from the cycle
                                        cycle_edges = [(u, a), (a, b), (b, c), (c, v)]
                                        ue, ve = random.choice(cycle_edges)
                                        adj[ue][ve] = False
                                        adj[ve][ue] = False
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    break
            
            # Check alpha
            if time.time() - start_time > _DEADLINE - _SAFETY_MARGIN:
                break
            
            # Check if alpha < 9
            # Use the branch and bound
            def has_alpha_ge_9(adj_local, n_local):
                def branch(remaining, size):
                    if size == 9:
                        return True
                    if len(remaining) < 9 - size:
                        return False
                    if size + len(remaining) < 9:
                        return False
                    # Pick vertex with highest degree in remaining
                    best_v = remaining[0]
                    best_deg = -1
                    for v in remaining:
                        deg = sum(adj_local[v][u] for u in remaining)
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                    
                    # Include best_v
                    new_rem = [u for u in remaining if u != best_v and not adj_local[best_v][u]]
                    if branch(new_rem, size + 1):
                        return True
                    # Exclude best_v
                    new_rem = [u for u in remaining if u != best_v]
                    if branch(new_rem, size):
                        return True
                    return False
                
                return branch(list(range(n_local)), 0)
            
            if not has_alpha_ge_9(adj, n_target):
                # Valid! Extract edges
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
    
    # Fallback: return the known good 28-vertex construction
    return {'n': best_n, 'edges': best_edges}
