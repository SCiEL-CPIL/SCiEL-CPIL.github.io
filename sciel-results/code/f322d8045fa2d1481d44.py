import math
import time
import numpy as np

def _jc_distance(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    start_time = time.time()
    deadline = 170.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n < 3:
        if n == 1: return f"({labels[0]});"
        if n == 2: return f"({labels[0]},{labels[1]});"
        
    # 1. Distance Matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            D[i, j] = d
            D[j, i] = d
            
    # 2. NJ Initial Tree
    # Returns a graph structure
    def nj_graph(dist, n):
        # Returns: 
        # nodes: list of node ids (0..n-1 leaves, n..2n-4 internal)
        # edges: list of (u, v)
        # active: list of active node ids
        # trees: dict node_id -> newick_string (for final output)
        
        active = list(range(n))
        trees = {i: labels[i] for i in range(n)}
        d = dist.copy()
        next_id = n
        edges = set() # Store undirected edges as frozenset or sorted tuple
        
        while len(active) > 2:
            m = len(active)
            r = [0.0] * m
            for i in range(m):
                for j in range(m):
                    if i != j:
                        r[i] += d[active[i], active[j]]
                        
            best_q = float('inf')
            best_i, best_j = -1, -1
            for i in range(m):
                for j in range(i + 1, m):
                    q = (m - 2) * d[active[i], active[j]] - r[i] - r[j]
                    if q < best_q:
                        best_q = q
                        best_i, best_j = i, j
                        
            ai = active[best_i]
            aj = active[best_j]
            
            d_new_ai = 0.5 * d[ai, aj] + (r[best_i] / (2 * (m - 2)))
            d_new_aj = d[ai, aj] - d_new_ai
            if d_new_ai < 0: d_new_ai = 0
            if d_new_aj < 0: d_new_aj = 0
            
            new_node = next_id
            next_id += 1
            trees[new_node] = f"({trees[ai]}:{d_new_ai:.4f},{trees[aj]}:{d_new_aj:.4f})"
            
            # Add edges
            edges.add((new_node, ai))
            edges.add((new_node, aj))
            
            # Update distances
            for k in active:
                if k != ai and k != aj:
                    d[new_node, k] = 0.5 * (d[ai, k] + d[aj, k] - d[ai, aj])
                    d[k, new_node] = d[new_node, k]
                    
            active = [x for x in active if x not in (ai, aj)]
            active.append(new_node)
            
        a, b = active[0], active[1]
        edges.add((a, b))
        # The last edge has no length assigned in this simple NJ, 
        # but for topology it's fine. We will re-estimate lengths later.
        
        # Build adjacency list
        adj = {i: set() for i in range(next_id)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        return adj, trees, next_id

    adj, trees, num_nodes = nj_graph(D, n)
    
    # 3. NNI Optimization
    # We need to evaluate topologies.
    # We will use Sum of Squared Residuals (SSR) with LS branch lengths.
    
    def get_path_lengths(adj, num_nodes, leaves):
        # This function is not needed directly, we need LS branch lengths.
        pass

    def compute_ls_branch_lengths(adj, num_nodes, leaves, D):
        # Solve for branch lengths
        # Variables: length of each edge
        # Equations: sum of lengths on path(i,j) = D[i,j]
        
        # Edges list
        edges = []
        edge_to_idx = {}
        idx = 0
        for u in range(num_nodes):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
                    edge_to_idx[(u, v)] = idx
                    edge_to_idx[(v, u)] = idx
                    idx += 1
                    
        E = len(edges)
        L = len(leaves)
        
        # A matrix: L*(L-1)/2 x E
        # We only need pairs of leaves
        pairs = []
        for i in range(L):
            for j in range(i + 1, L):
                pairs.append((leaves[i], leaves[j]))
                
        M = len(pairs)
        A = np.zeros((M, E))
        b = np.zeros(M)
        
        # For each pair, find path
        # Since n is small, we can do BFS/DFS for each pair
        for k, (u, v) in enumerate(pairs):
            # Find path from u to v
            # BFS
            visited = {u}
            queue = [(u, [u])]
            path = None
            while queue:
                node, path_so_far = queue.pop(0)
                if node == v:
                    path = path_so_far
                    break
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path_so_far + [neighbor]))
                        
            if path:
                for i in range(len(path) - 1):
                    e1, e2 = path[i], path[i+1]
                    if (e1, e2) in edge_to_idx:
                        A[k, edge_to_idx[(e1, e2)]] = 1
                    elif (e2, e1) in edge_to_idx:
                        A[k, edge_to_idx[(e2, e1)]] = 1
                b[k] = D[u, v]
                
        # Solve A l = b using least squares
        if E == 0:
            return {}
            
        try:
            # l, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            l, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # Clip negative lengths to 0?
            # For SSR, we use the raw solution.
            return {edges[i]: max(0, l[i]) for i in range(E)}
        except:
            return {e: 0.0 for e in edges}

    def compute_ssr(adj, num_nodes, leaves, D, branch_lengths):
        # Compute SSR
        L = len(leaves)
        pairs = []
        for i in range(L):
            for j in range(i + 1, L):
                pairs.append((leaves[i], leaves[j]))
                
        ssr = 0.0
        for u, v in pairs:
            # Find path
            visited = {u}
            queue = [(u, [u])]
            path = None
            while queue:
                node, path_so_far = queue.pop(0)
                if node == v:
                    path = path_so_far
                    break
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path_so_far + [neighbor]))
                        
            if path:
                d_path = 0.0
                for i in range(len(path) - 1):
                    e1, e2 = path[i], path[i+1]
                    if (e1, e2) in branch_lengths:
                        d_path += branch_lengths[(e1, e2)]
                    elif (e2, e1) in branch_lengths:
                        d_path += branch_lengths[(e2, e1)]
                ssr += (D[u, v] - d_path) ** 2
                
        return ssr

    leaves = list(range(n))
    
    # Initial lengths
    bl = compute_ls_branch_lengths(adj, num_nodes, leaves, D)
    current_ssr = compute_ssr(adj, num_nodes, leaves, D, bl)
    
    # NNI Loop
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_ssr = current_ssr
        best_move = None
        
        # Find all internal edges
        # An internal edge connects two nodes with degree 3 (or more if we consider the tree)
        # In an unrooted binary tree, all internal nodes have degree 3.
        internal_edges = []
        for u in range(num_nodes):
            if len(adj[u]) == 3:
                for v in adj[u]:
                    if u < v and len(adj[v]) == 3:
                        internal_edges.append((u, v))
                        
        for u, v in internal_edges:
            # Neighbors of u: v, a, b
            # Neighbors of v: u, c, d
            neighbors_u = list(adj[u])
            neighbors_v = list(adj[v])
            
            a = neighbors_u[0]
            b = neighbors_u[1]
            if neighbors_u[0] == v:
                a, b = neighbors_u[1], neighbors_u[2]
            elif neighbors_u[1] == v:
                a, b = neighbors_u[0], neighbors_u[2]
            else:
                a, b = neighbors_u[0], neighbors_u[1]
                
            c = neighbors_v[0]
            d = neighbors_v[1]
            if neighbors_v[0] == u:
                c, d = neighbors_v[1], neighbors_v[2]
            elif neighbors_v[1] == u:
                c, d = neighbors_v[0], neighbors_v[2]
            else:
                c, d = neighbors_v[0], neighbors_v[1]
                
            # Move 1: u-c, v-a
            # Remove u-a, v-c
            # Add u-c, v-a
            # Check if this is a valid NNI (a != c, etc.)
            if a != c and a != d and b != c and b != d:
                # Create new adj
                new_adj = {i: set(adj[i]) for i in range(num_nodes)}
                new_adj[u].remove(a)
                new_adj[a].remove(u)
                new_adj[v].remove(c)
                new_adj[c].remove(v)
                
                new_adj[u].add(c)
                new_adj[c].add(u)
                new_adj[v].add(a)
                new_adj[a].add(v)
                
                # Evaluate
                bl_new = compute_ls_branch_lengths(new_adj, num_nodes, leaves, D)
                ssr_new = compute_ssr(new_adj, num_nodes, leaves, D, bl_new)
                
                if ssr_new < best_ssr:
                    best_ssr = ssr_new
                    best_move = (new_adj, bl_new, "1")
                    
            # Move 2: u-d, v-b
            if a != d and a != c and b != d and b != c: # Same check
                new_adj = {i: set(adj[i]) for i in range(num_nodes)}
                new_adj[u].remove(b)
                new_adj[b].remove(u)
                new_adj[v].remove(d)
                new_adj[d].remove(v)
                
                new_adj[u].add(d)
                new_adj[d].add(u)
                new_adj[v].add(b)
                new_adj[b].add(v)
                
                bl_new = compute_ls_branch_lengths(new_adj, num_nodes, leaves, D)
                ssr_new = compute_ssr(new_adj, num_nodes, leaves, D, bl_new)
                
                if ssr_new < best_ssr:
                    best_ssr = ssr_new
                    best_move = (new_adj, bl_new, "2")
                    
        if best_move:
            adj, bl, _ = best_move
            current_ssr = best_ssr
            improved = True
            
    # Convert back to Newick
    # We need to reconstruct the Newick string from the graph and branch lengths.
    # We can use a DFS from an arbitrary leaf.
    
    def graph_to_newick(adj, num_nodes, leaves, branch_lengths):
        # Pick a leaf as root
        root = leaves[0]
        parent = {root: -1}
        visited = {root}
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
                    
        # Build children map
        children = {i: [] for i in range(num_nodes)}
        for node in range(1, num_nodes):
            p = parent[node]
            if p != -1:
                children[p].append(node)
                
        def dfs(node):
            if node in leaves:
                # Leaf
                # Find the edge to parent
                p = parent[node]
                if p != -1:
                    if (node, p) in branch_lengths:
                        l = branch_lengths[(node, p)]
                    else:
                        l = branch_lengths.get((p, node), 0.0)
                    return f"{labels[node]}:{l:.4f}"
                else:
                    return labels[node]
            else:
                # Internal node
                # Find edge to parent
                p = parent[node]
                if p != -1:
                    if (node, p) in branch_lengths:
                        l = branch_lengths[(node, p)]
                    else:
                        l = branch_lengths.get((p, node), 0.0)
                    suffix = f":{l:.4f}"
                else:
                    suffix = ""
                    
                child_strs = []
                for child in children[node]:
                    child_strs.append(dfs(child))
                    
                return "(" + ",".join(child_strs) + ")" + suffix
                
        return dfs(root)

    final_newick = graph_to_newick(adj, num_nodes, leaves, bl)
    return final_newick
