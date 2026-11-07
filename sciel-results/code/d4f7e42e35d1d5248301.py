"""Packing trees T1..T20 into K20 - edge-disjoint tree packing."""
import time
from random import Random

def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def _build_adjacency(tree_edges, k):
    """Build adjacency list for a tree with k vertices."""
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _root_tree(adj, k, root=0):
    """Root the tree at given vertex, return parent and order (BFS/DFS order)."""
    parent = [-1] * k
    order = []
    visited = [False] * k
    stack = [root]
    visited[root] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                stack.append(v)
    return parent, order

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    n = 20
    trees = _instance_trees(n, 611)
    
    # Precompute adjacency and rooted structure for each tree
    tree_adjs = []
    tree_parents = []
    tree_orders = []
    for k in range(1, n + 1):
        if k == 1:
            tree_adjs.append([])
            tree_parents.append([-1])
            tree_orders.append([0])
        else:
            adj = _build_adjacency(trees[k-1], k)
            tree_adjs.append(adj)
            parent, order = _root_tree(adj, k, root=0)
            tree_parents.append(parent)
            tree_orders.append(order)
    
    # Track used edges in K20 (frozenset of frozensets)
    used_edges = set()
    result = [None] * n
    
    # Embed trees from largest to smallest (hardest first)
    # T20 first, then T19, ..., T1
    order_k = list(range(n - 1, 0, -1))  # k = 19, 18, ..., 1 (0-indexed: tree index k-1)
    
    # For each tree, we need to find an edge-disjoint embedding
    # Use DFS with backtracking, trying vertices in a smart order
    
    def find_embedding(k_idx, adj, parent, order, used_edges, deadline):
        """Find edge-disjoint embedding of tree k_idx into K20.
        Returns list of 20 host vertices or None."""
        k = k_idx + 1
        if k == 1:
            # Single vertex, any unused... actually no edge constraint
            return [0]  # vertex 0
        
        # We need to assign each tree vertex a host vertex
        # Constraint: for each edge (u,v) in tree, (host[u], host[v]) not in used_edges
        # and host[u] != host[v]
        
        # Use the rooted tree structure: assign root first, then children
        # This ensures we can check edge constraints incrementally
        
        # Order: root, then in BFS order
        # parent[0] = -1 for root
        root_tree_vertex = 0
        # Get children of each vertex
        children = [[] for _ in range(k)]
        for v in range(k):
            if parent[v] != -1:
                children[parent[v]].append(v)
        
        # Try to assign using backtracking
        # State: assignment array, position in order
        assignment = [None] * k
        used_local = set()  # edges used by this embedding (for quick check)
        
        # Symmetry breaking: fix root to vertex 0
        # Actually, let's try multiple root placements but with time limit
        
        # Build the processing order (BFS from root)
        proc_order = [root_tree_vertex]
        for v in proc_order[:]:
            for c in children[v]:
                proc_order.append(c)
        
        def dfs(pos):
            if time.time() > deadline:
                return False
            if pos == len(proc_order):
                return True
            tv = proc_order[pos]
            pv = parent[tv]
            
            if pv == -1:
                # Root: try each vertex 0..19
                candidates = list(range(20))
            else:
                # Must be different from parent's host, and edge not used
                ph = assignment[pv]
                candidates = [v for v in range(20) if v != ph and frozenset((ph, v)) not in used_edges and frozenset((ph, v)) not in used_local]
            
            # Try candidates in order (could randomize for variety)
            for host in candidates:
                assignment[tv] = host
                # Check if edge to parent is valid (already checked in candidates)
                # Add edge to used_local
                if pv != -1:
                    edge = frozenset((pv, tv))  # tree edge, not host edge
                    host_edge = frozenset((assignment[pv], host))
                    used_local.add(host_edge)
                
                if dfs(pos + 1):
                    return True
                
                if pv != -1:
                    used_local.discard(frozenset((assignment[pv], host)))
                assignment[tv] = None
            
            return False
        
        # Try with root fixed to 0 first (symmetry breaking)
        # Then try other roots if time permits
        for root_host in range(20):
            if time.time() > deadline:
                break
            assignment = [None] * k
            used_local = set()
            
            # Override: force root to root_host
            # We need a modified DFS
            def dfs_root_fixed(pos, root_host_val):
                if time.time() > deadline:
                    return False
                if pos == len(proc_order):
                    return True
                tv = proc_order[pos]
                pv = parent[tv]
                
                if pv == -1:
                    if tv == root_tree_vertex:
                        host = root_host_val
                        assignment[tv] = host
                        if dfs_root_fixed(pos + 1, root_host_val):
                            return True
                        assignment[tv] = None
                        return False
                    else:
                        # Shouldn't happen
                        return False
                else:
                    ph = assignment[pv]
                    candidates = [v for v in range(20) if v != ph and frozenset((ph, v)) not in used_edges and frozenset((ph, v)) not in used_local]
                    
                    for host in candidates:
                        assignment[tv] = host
                        host_edge = frozenset((ph, host))
                        used_local.add(host_edge)
                        
                        if dfs_root_fixed(pos + 1, root_host_val):
                            return True
                        
                        used_local.discard(host_edge)
                        assignment[tv] = None
                    
                    return False
            
            if dfs_root_fixed(0, root_host):
                return assignment
        
        return None
    
    # Embed in order from largest to smallest
    success_count = 0
    for k_idx in order_k:
        if time.time() > deadline:
            break
        k = k_idx + 1
        adj = tree_adjs[k_idx]
        parent = tree_parents[k_idx]
        
        embedding = find_embedding(k_idx, adj, parent, tree_orders[k_idx], used_edges, deadline)
        
        if embedding is not None:
            # Verify and add edges
            valid = True
            for u, v in trees[k_idx]:
                hu, hv = embedding[u], embedding[v]
                if hu == hv:
                    valid = False
                    break
                edge = frozenset((hu, hv))
                if edge in used_edges:
                    valid = False
                    break
            if valid:
                for u, v in trees[k_idx]:
                    used_edges.add(frozenset((embedding[u], embedding[v])))
                result[k_idx] = [int(embedding[i]) for i in range(k)]
                success_count += 1
            else:
                result[k_idx] = None
        else:
            result[k_idx] = None
    
    # Ensure all values are Python ints
    for i in range(n):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
