from random import Random
import time

def instance_trees(n=20, seed=611):
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    trees = instance_trees()
    # trees[k-1] is tree T_k with k vertices, k-1 edges
    # We need to embed each tree T_k into K20 (vertices 0..19)
    # Embedding: list of k ints, position i = host vertex for tree vertex i
    
    n = 20
    # Total edges available: 190
    # Total tree edges: sum(k-1 for k=1..20) = 190
    # So we need perfect packing
    
    # Strategy: process trees from largest to smallest
    # T20 is a spanning tree (uses all 20 vertices)
    # For T20, fix vertex 0 -> host 0 to break symmetry
    
    # Use backtracking with MRV/LCV and randomized restarts
    
    best_result = None
    best_edges = 0
    
    def try_packing(seed_offset):
        rng = Random(42 + seed_offset)
        used_edges = set()  # set of frozenset({u,v}) for used edges
        embeddings = [None] * 20  # embeddings[k-1] = list of k host vertices
        
        # Process trees in order: T20, T19, T18, ..., T1
        # T20 must use all 20 vertices
        # For T20, fix tree vertex 0 -> host 0
        
        # Build adjacency for each tree
        tree_adj = []
        for k in range(1, n + 1):
            adj = [[] for _ in range(k)]
            for u, v in trees[k - 1]:
                adj[u].append(v)
                adj[v].append(u)
            tree_adj.append(adj)
        
        # Order: process larger trees first
        order = list(range(n - 1, -1, -1))  # index k-1 for tree T_k, so n-1 down to 0
        
        def backtrack(idx):
            nonlocal used_edges, embeddings
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True
            
            tree_idx = order[idx]  # 0-indexed, tree_idx+1 is the tree size
            k = tree_idx + 1  # number of vertices in this tree
            adj = tree_adj[tree_idx]
            
            if k == 1:
                # T1: single vertex, no edges. Any unused vertex works.
                # Actually T1 has no edges, so any vertex is fine
                # But we need it to be a valid vertex 0..19
                # Since no edges, just pick any vertex
                # But it must be distinct from... well, it's just one vertex
                # It doesn't conflict with any edges
                embeddings[tree_idx] = [0]  # just pick vertex 0, doesn't matter
                return backtrack(idx + 1)
            
            # For tree with k vertices, find an embedding (injective map to 0..19)
            # such that all edges are unused
            
            # Use backtracking within the embedding search
            # MRV: assign vertices in order of degree (highest first)
            # LCV: prefer host vertices with more available neighbors
            
            # Get vertex ordering by degree (descending)
            deg = [len(adj[v]) for v in range(k)]
            vertex_order = sorted(range(k), key=lambda v: -deg[v])
            
            # For T20 (k=20), fix vertex 0 to host 0
            # Actually, let's use symmetry breaking: fix the first vertex in order to a specific host
            
            def embed_backtrack(vpos, assignment, available_hosts):
                nonlocal used_edges
                if time.time() > deadline:
                    return False
                if vpos == k:
                    # Check all edges are valid (should be by construction)
                    # Record embedding
                    emb = [0] * k
                    for tv, hv in assignment:
                        emb[tv] = hv
                    embeddings[tree_idx] = emb
                    return True
                
                tv = vertex_order[vpos]
                
                # For symmetry breaking on T20: if this is the first vertex and k==20, fix to host 0
                if k == 20 and vpos == 0:
                    if 0 not in available_hosts:
                        return False
                    new_avail = available_hosts - {0}
                    # Check if any edge from tv to already-assigned vertices conflicts
                    # No vertices assigned yet, so no edge check needed
                    if embed_backtrack(vpos + 1, assignment + [(tv, 0)], new_avail):
                        return True
                    return False
                
                # Try each available host for this tree vertex
                # LCV: sort hosts by how many of their neighbors are still available
                hosts = list(available_hosts)
                # Sort by number of available neighbors (more is better)
                def host_score(h):
                    count = 0
                    for nb in adj[tv]:
                        if nb in assignment_dict and assignment_dict[nb] in available_hosts:
                            count += 1
                    return count
                
                # Actually, let's just try hosts in a randomized order for diversity
                rng.shuffle(hosts)
                
                assignment_dict = dict(assignment)
                
                for h in hosts:
                    # Check: for each neighbor nb of tv that is already assigned,
                    # the edge (h, assignment_dict[nb]) must not be in used_edges
                    conflict = False
                    for nb in adj[tv]:
                        if nb in assignment_dict:
                            edge = frozenset((h, assignment_dict[nb]))
                            if edge in used_edges:
                                conflict = True
                                break
                    if conflict:
                        continue
                    
                    # Also check degree constraint: h's total degree in used_edges + new edges
                    # must not exceed 19. But since we're building incrementally,
                    # and total is exactly 190, we need to be careful.
                    # For now, just check edge conflicts.
                    
                    new_avail = available_hosts - {h}
                    if embed_backtrack(vpos + 1, assignment + [(tv, h)], new_avail):
                        return True
                
                return False
            
            avail = set(range(n))
            if embed_backtrack(0, [], avail):
                # Record used edges
                for u, v in trees[tree_idx]:
                    edge = frozenset((embeddings[tree_idx][u], embeddings[tree_idx][v]))
                    used_edges.add(edge)
                return backtrack(idx + 1)
            else:
                return False
        
        if backtrack(0):
            # Verify and compute score
            total_edges = 0
            for k in range(1, n + 1):
                emb = embeddings[k - 1]
                if emb is not None:
                    total_edges += k - 1
            return embeddings, total_edges
        return None, 0
    
    # Try with different seeds
    for attempt in range(500):
        if time.time() > deadline:
            break
        result, edges = try_packing(attempt)
        if result is not None:
            if edges > best_edges:
                best_edges = edges
                best_result = result
            if edges == 190:
                break
    
    if best_result is not None:
        # Convert to proper format
        out = []
        for k in range(1, n + 1):
            emb = best_result[k - 1]
            if emb is None:
                out.append(None)
            else:
                out.append([int(x) for x in emb])
        return out
    else:
        # Fallback: just T20 as identity
        return [None] * 19 + [list(range(20))]
