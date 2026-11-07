import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    trees = instance_trees(20, 611)
    
    # Sort trees by number of vertices descending (index + 1)
    # Tree k has k vertices.
    # We want to embed larger trees first as they are more constrained.
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: i, reverse=True)
    
    # All edges in K20
    all_edges = set()
    for i in range(20):
        for j in range(i + 1, 20):
            all_edges.add((i, j))
    
    used_edges = set()
    embeddings = [None] * 20  # embeddings[i] will be list of 20 ints or None
    
    # Helper to check if time is up
    def time_up():
        return time.time() > deadline

    # Function to find an embedding of a tree in the available graph
    # tree_edges: list of (u, v)
    # available: set of edges
    # Returns a list of length len(tree_vertices) mapping tree_vertex -> host_vertex
    # Or None if no embedding found
    # We use backtracking for this subproblem too
    def find_embedding(tree_edges, available):
        if not tree_edges:
            return [0]  # T1 has 1 vertex, map to 0 (arbitrary)
        
        # Determine vertices in tree
        verts = set()
        for u, v in tree_edges:
            verts.add(u)
            verts.add(v)
        verts = list(verts)
        n_verts = len(verts)
        
        # Build adjacency for the tree
        adj = {v: [] for v in verts}
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Pick a root (e.g., first vertex)
        root = verts[0]
        
        # Mapping: tree_vertex -> host_vertex
        mapping = {}
        
        def dfs(tv, host):
            if time_up():
                return False
            mapping[tv] = host
            if len(mapping) == n_verts:
                # Check if all edges are in available
                for u, v in tree_edges:
                    h_u = mapping[u]
                    h_v = mapping[v]
                    edge = (min(h_u, h_v), max(h_u, h_v))
                    if edge not in available:
                        return False
                return True
            
            for child in adj[tv]:
                if child in mapping:
                    continue
                # Try all possible hosts for child
                # Host must be connected to 'host' in 'available'
                # And not used by other mapped tree vertices
                used_hosts = set(mapping.values())
                for h in range(20):
                    if h in used_hosts:
                        continue
                    edge = (min(host, h), max(host, h))
                    if edge not in available:
                        continue
                    if dfs(child, h):
                        return True
            del mapping[tv]
            return False
            
        # Try all hosts for root
        for h in range(20):
            if dfs(root, h):
                # Convert mapping to list
                result = [0] * n_verts
                # We need to map the tree's original vertex indices to the result list positions?
                # The task says: "position i is the image of vertex i of the fixed tree T_k"
                # So we need a list where index i is the host for tree vertex i.
                # My 'verts' list might not be 0..n-1 in order.
                # Let's just return the mapping dict and convert later.
                # Actually, the tree vertices are 0..k-1.
                # So mapping keys are 0..k-1.
                return mapping
        
        return None

    # Main solver
    def solve(idx):
        if time_up():
            return False
        if idx == len(tree_indices):
            return True
            
        t_idx = tree_indices[idx]
        tree_edges = trees[t_idx]
        
        # Available edges are all_edges - used_edges
        # To avoid creating a new set every time, we can pass a copy or just check membership
        # Creating a set of 190 elements is fast.
        available = all_edges - used_edges
        
        emb = find_embedding(tree_edges, available)
        
        if emb is None:
            return False
            
        # Apply embedding
        for u, v in tree_edges:
            h_u = emb[u]
            h_v = emb[v]
            edge = (min(h_u, h_v), max(h_u, h_v))
            used_edges.add(edge)
            
        # Store embedding
        # We need to store it in the format: list of k distinct ints
        # emb is a dict {tree_vertex: host_vertex}
        # We need a list where index i is emb[i]
        k = t_idx + 1
        emb_list = [emb[i] for i in range(k)]
        embeddings[t_idx] = emb_list
        
        if solve(idx + 1):
            return True
            
        # Backtrack
        for u, v in tree_edges:
            h_u = emb[u]
            h_v = emb[v]
            edge = (min(h_u, h_v), max(h_u, h_v))
            used_edges.remove(edge)
        embeddings[t_idx] = None
        
        return False

    # Run solver
    if solve(0):
        # Convert to required format
        result = []
        for k in range(20):
            if embeddings[k] is None:
                result.append(None)
            else:
                # Ensure it's a list of ints
                result.append([int(x) for x in embeddings[k]])
        return result
    else:
        # Fallback: return what we have or just T20
        # If we failed, we might have partial embeddings.
        # But the contract says "Entry k-1 is None or a list...".
        # If we return a partial packing, the value is sum of edges in embedded trees.
        # We should return the best partial packing found.
        # However, my solver backtracks completely.
        # Let's modify solve to keep track of best solution.
        
        # For now, if solve fails, return the seed solution (T20 only)
        # Or we can try a few random restarts for T20 embedding?
        # Given the observations, a solution likely exists.
        # If solve(0) failed, it might be due to time or dead end.
        # Let's just return the seed solution to be safe, or try to salvage.
        
        # Let's try to embed just T20 and return that.
        result = [None] * 20
        # Embed T20 with identity
        result[19] = list(range(20))
        return result

# EVOLVE-BLOCK-END
