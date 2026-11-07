import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Build adjacency info
    # Edge list: top cycle, bottom cycle, matching
    # We need to identify which edges are which type
    
    # Let's build the graph structure
    # Top cycle: vertices 0..m-1
    # Bottom cycle: vertices m..2m-1
    # Matching: (i, i+m) for i in 0..m-1
    
    # Identify edge types
    edge_info = []
    for (u, v) in edges:
        if u < m and v < m:
            # Top cycle edge
            # Determine which position in the cycle
            if v == (u + 1) % m:
                pos = u
            else:
                pos = (u - 1) % m
            edge_info.append(('top', pos))
        elif u >= m and v >= m:
            # Bottom cycle edge
            u2 = u - m
            v2 = v - m
            if v2 == (u2 + 1) % m:
                pos = u2
            else:
                pos = (u2 - 1) % m
            edge_info.append(('bottom', pos))
        else:
            # Matching edge
            if u < m:
                pos = u
            else:
                pos = u - m
            edge_info.append(('match', pos))
    
    # Number of edges
    E = len(edges)
    
    # For switching classes, we fix a spanning tree and vary the rest
    # A spanning tree of the prism: 
    # - Top cycle: m-1 edges (break one edge, say the last one)
    # - Bottom cycle: m-1 edges (break one edge)
    # - Matching: all m edges
    # Total: (m-1) + (m-1) + m = 3m - 2 edges
    # But we need 2m-1 edges for a spanning tree. 3m-2 vs 2m-1: for m>=3, 3m-2 > 2m-1
    # So we have more than needed. Let me pick a specific spanning tree.
    
    # Spanning tree: 
    # - Top path: (0,1), (1,2), ..., (m-2, m-1) -> m-1 edges
    # - Bottom path: (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1) -> m-1 edges
    # - Matching: (0,m), (1,m+1), ..., (m-2, 2m-2) -> m-1 edges
    # Total: 3(m-1) = 3m-3 edges. Need 2m-1. For m=4: 9 vs 7. Still too many.
    
    # Let me just use a simpler approach: enumerate all 2^{m+1} switching classes
    # by fixing signs of a spanning tree and varying the m+1 remaining edges.
    
    # Actually, let me just do a direct approach: 
    # The spectral radius depends only on the switching class.
    # I'll enumerate all 2^{m+1} classes by choosing signs for m+1 specific edges
    # and determining the rest via switching.
    
    # Simpler: just enumerate all possible sign assignments for the m+1 "independent"
    # edges, compute the full sign assignment, and evaluate.
    
    # Let me identify which edges are in a spanning tree and which are not.
    
    # Build a spanning tree using BFS from vertex 0
    from collections import deque
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # BFS to find spanning tree
    visited = [False] * n
    parent_edge = [-1] * n
    visited[0] = True
    queue = deque([0])
    tree_edges = set()
    
    while queue:
        u = queue.popleft()
        for v, eidx in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent_edge[v] = eidx
                tree_edges.add(eidx)
                queue.append(v)
    
    # Non-tree edges
    non_tree_edges = [i for i in range(E) if i not in tree_edges]
    num_non_tree = len(non_tree_edges)
    
    # num_non_tree should be m+1 for a prism graph
    # E = 3m, tree edges = 2m-1, so non-tree = 3m - (2m-1) = m+1. Correct.
    
    # For each assignment of signs to non-tree edges, determine the full sign assignment
    # by switching. We fix vertex 0 to have switching value +1, and determine others.
    
    # Actually, the switching operation is: for each vertex v, multiply all incident edge signs
    # by a factor of ±1. The switching class is determined by the product of signs around
    # each cycle.
    
    # Let me use a different approach: directly enumerate all 2^{m+1} switching classes
    # by assigning signs to the non-tree edges, then computing the tree edge signs
    # such that the cycle products are consistent.
    
    # Actually, I think the cleanest approach is:
    # 1. Assign arbitrary signs to the non-tree edges (2^{m+1} possibilities)
    # 2. For the tree edges, the signs are determined up to switching
    # 3. Fix the switching by setting all vertex switching factors to +1
    # 4. The tree edge signs are then determined by the requirement that
    #    the product around each fundamental cycle equals the product of the
    #    non-tree edge signs in that cycle.
    
    # This is getting complex. Let me just do a brute force over all 2^{3m} sign
    # assignments for small m, but that's too many for m=9 (2^27).
    
    # Better: use the fact that switching classes are 2^{m+1}. I'll pick a reference
    # sign assignment and generate all switching classes from it.
    
    # Reference: all +1
    # Switching: for each subset S of vertices, flip signs of edges incident to vertices in S
    # But this generates 2^{2m} assignments, with 2^{2m-1} being switching-equivalent
    # to all +1... no, that's not right either.
    
    # Let me think differently. The signed adjacency matrix A has the property that
    # A^2 is determined by the graph structure and the products of signs around cycles.
    # For a bipartite-like structure, this might help.
    
    # Actually, for the purpose of this problem, let me just do a smart random search
    # combined with local optimization, with a deadline guard.
    
    # But the strategy says to use QUBO with trace(A^4) as surrogate. Let me try that.
    # trace(A^4) = sum of 4-cycles with sign products. Minimizing this should correlate
    # with minimizing spectral radius.
    
    # Actually, let me just enumerate all 2^{m+1} switching classes properly.
    
    # Method: 
    # 1. Pick a spanning tree T
    # 2. The m+1 non-tree edges each correspond to a fundamental cycle
    # 3. The product of signs around each fundamental cycle is invariant under switching
    # 4. So the switching class is determined by these m+1 cycle products
    # 5. For each of the 2^{m+1} assignments of cycle products, we can compute the
    #    sign assignment (up to switching) and then pick a representative.
    
    # To get a representative: fix the sign of one tree edge (say the first one) to +1,
    # then determine all other tree edge signs by propagating, and the non-tree edge
    # signs are determined by the cycle products.
    
    # Let me implement this.
    
    # First, build the fundamental cycles
    # For each non-tree edge (u,v), the fundamental cycle is the path from u to v in T plus the edge (u,v)
    
    def get_path_in_tree(u, v, parent, parent_edge, adj):
        """Get path from u to v in the tree, returning list of edge indices."""
        # Find LCA or just do BFS
        # Since it's a tree, there's a unique path
        # Do BFS from u
        visited = {u}
        parent_map = {}
        queue = deque([(u, [])])
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for neighbor, eidx in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = (node, eidx)
                    queue.append((neighbor, path + [eidx]))
        return None
    
    # Build fundamental cycles
    fundamental_cycles = []
    for eidx in non_tree_edges:
        u, v = edges[eidx]
        path = get_path_in_tree(u, v, parent_edge, parent_edge, adj)
        if path is not None:
            # The cycle is: u -> ... -> v (via path) -> u (via eidx)
            cycle_edges = path + [eidx]
            fundamental_cycles.append(cycle_edges)
    
    # Now enumerate all 2^{m+1} switching classes
    best_radius = float('inf')
    best_signs = None
    num_classes = 1 << num_non_tree
    
    # Precompute: for each class, the cycle products
    # Cycle product for fundamental cycle j is the product of signs on its edges
    # This must equal the assigned value for that cycle
    
    # For a given assignment of cycle products c_0, c_1, ..., c_{m} (each ±1),
    # we need to find signs s_e for all edges such that:
    #   product_{e in cycle_j} s_e = c_j for each j
    
    # This is a system of linear equations over GF(2) (with +1=0, -1=1)
    
    # Convert to GF(2): let x_e = 0 if s_e = +1, x_e = 1 if s_e = -1
    # Then sum_{e in cycle_j} x_e = 0 if c_j = +1, 1 if c_j = -1
    
    # Build the cycle-edge incidence matrix
    num_edges = E
    num_cycles = num_non_tree
    
    # Incidence matrix M: num_cycles x num_edges, M[j,e] = 1 if edge e is in cycle j
    M = np.zeros((num_cycles, num_edges), dtype=np.int8)
    for j, cycle_edges in enumerate(fundamental_cycles):
        for e in cycle_edges:
            M[j, e] = 1
    
    # For each of the 2^{num_cycles} assignments of cycle products:
    #   b_j = 0 if c_j = +1, 1 if c_j = -1
    #   Solve M x = b over GF(2)
    #   If consistent, get a solution x, convert to signs
    
    # The system is consistent for any b because the cycles span the cycle space.
    # There are 2^{num_edges - rank(M)} solutions, but we just need one.
    
    # Use Gaussian elimination over GF(2)
    
    def solve_gf2(M, b):
        """Solve M x = b over GF(2). Returns a solution or None if inconsistent."""
        M = M.copy()
        b = b.copy()
        n_rows = len(b)
        n_cols = M.shape[1]
        
        # Augmented matrix
        A = np.hstack([M, b.reshape(-1, 1)])
        
        # Gaussian elimination
        rank = 0
        for col in range(n_cols):
            # Find pivot
            pivot_row = -1
            for row in range(rank, n_rows):
                if A[row, col] == 1:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            # Swap rows
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
            # Eliminate
            for row in range(n_rows):
                if row != rank and A[row, col] == 1:
                    A[row] = (A[row] + A[rank]) % 2
            rank += 1
            if rank == n_rows:
                break
        
        # Check consistency
        for row in range(rank, n_rows):
            if A[row, n_cols] == 1:
                return None
        
        # Get a solution (set free variables to 0)
        x = np.zeros(n_cols, dtype=np.int8)
        for i in range(rank):
            # Find the pivot column for row i
            for col in range(n_cols):
                if A[i, col] == 1:
                    x[col] = A[i, n_cols]
                    break
        
        return x
    
    # Enumerate all switching classes
    for mask in range(num_classes):
        if time.time() > deadline:
            break
        
        # Get cycle products from mask
        b = np.zeros(num_cycles, dtype=np.int8)
        for j in range(num_cycles):
            if (mask >> j) & 1:
                b[j] = 1  # cycle product = -1
            else:
                b[j] = 0  # cycle product = +1
        
        # Solve for edge signs
        x = solve_gf2(M, b)
        if x is None:
            continue
        
        # Convert to signs
        signs = np.ones(num_edges, dtype=np.int8)
        signs[x == 1] = -1
        
        # Compute spectral radius
        # Build signed adjacency matrix
        A_mat = np.zeros((n, n), dtype=np.float64)
        for eidx, (u, v) in enumerate(edges):
            s = signs[eidx]
            A_mat[u, v] = s
            A_mat[v, u] = s
        
        # Check if lift is connected and 3-regular
        # The lift is always 3-regular for a 3-regular base graph with any signs
        # Need to check connectivity of the lift
        
        # Build lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        
        lift_adj = [[] for _ in range(2 * n)]
        for eidx, (u, v) in enumerate(edges):
            s = signs[eidx]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # Check connectivity via BFS
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        if count != 2 * n:
            # Not connected, skip
            continue
        
        # Compute spectral radius
        eigenvalues = np.linalg.eigvalsh(A_mat)
        radius = np.max(np.abs(eigenvalues))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is not None:
        return best_signs
    else:
        # Fallback: all +1
        return np.ones(len(edges), dtype=np.int8)
