import time
import random
from itertools import permutations

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees(20, 611)
    
    # Result structure: list of 20 entries.
    # Entry k-1 is None or list of k distinct ints.
    result = [None] * 20
    
    # T20 (index 19) is fixed to identity.
    # This is a valid embedding if T20 is a tree on 20 vertices.
    # The generator produces a tree on k vertices.
    # We map tree vertex i to host vertex i.
    result[19] = list(range(20))
    
    # Track used edges in K20
    used_edges = set()
    for u, v in trees[19]:
        if u > v: u, v = v, u
        used_edges.add((u, v))
        
    # Function to embed a single tree using backtracking
    # tree_edges: list of (u, v) for vertices 0..k-1
    # k: number of vertices
    # used_edges: set of edges already used in K20
    # Returns: mapping list of length 20 (host vertices for tree verts 0..k-1, -1 for others) or None
    def embed_tree(tree_edges, k, used_edges, rng):
        if k == 1:
            # Any available vertex works, but must be distinct from others (none).
            # Just pick 0.
            mapping = [-1] * 20
            mapping[0] = 0
            return mapping
            
        if k == 2:
            # Find any free edge
            for u in range(20):
                for v in range(u+1, 20):
                    if (u, v) not in used_edges:
                        mapping = [-1] * 20
                        mapping[0] = u
                        mapping[1] = v
                        return mapping
            return None
            
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Order vertices for search: Root at highest degree, then DFS order
        # We will use a recursive search.
        # State: mapping array, set of used host vertices
        # To speed up, we order the tree vertices such that we assign high degree ones first?
        # Or just do a DFS from a root.
        
        root = max(range(k), key=lambda x: len(adj[x]))
        
        # We need to try different host vertices for the root.
        # To add randomness/restarts, we shuffle the order of host vertices tried.
        
        # Precompute list of free edges? No, just check on the fly.
        
        def is_edge_free(u, v):
            if u > v: u, v = v, u
            return (u, v) not in used_edges
            
        # We will use a standard backtracking search.
        # To make it faster, we can use a heuristic:
        # When assigning a tree vertex, prefer host vertices that have fewer free edges to unassigned tree neighbors?
        # Or simpler: just try all 20 vertices.
        
        # We'll try up to a certain number of steps or time.
        # Since K20 is dense, it should be fast.
        
        mapping = [-1] * k
        used_hosts = set()
        
        # Shuffle host vertices for the root to allow restarts
        host_order = list(range(20))
        rng.shuffle(host_order)
        
        # We'll use a counter to limit backtracking depth/steps if needed, 
        # but for k<=19 in K20, it should be very fast.
        
        def dfs(v, parent):
            # v is tree vertex, parent is tree parent
            if time.time() > deadline:
                return False
                
            # If v is not mapped
            if mapping[v] == -1:
                # Try each host vertex in host_order
                for h in host_order:
                    if h in used_hosts:
                        continue
                        
                    # Check if edges to mapped neighbors are free
                    ok = True
                    for nb in adj[v]:
                        if mapping[nb] != -1:
                            if not is_edge_free(h, mapping[nb]):
                                ok = False
                                break
                    if not ok:
                        continue
                        
                    # Assign
                    mapping[v] = h
                    used_hosts.add(h)
                    
                    # Recurse to neighbors
                    success = True
                    for nb in adj[v]:
                        if mapping[nb] == -1:
                            if not dfs(nb, v):
                                success = False
                                break
                                
                    if success:
                        return True
                    else:
                        # Backtrack
                        used_hosts.remove(h)
                        mapping[v] = -1
                        
            return False
            
        # Try to start DFS from root
        # We need to handle the fact that `dfs` relies on `mapping` being initialized.
        # We set mapping[root] implicitly? No, dfs checks if mapping[v] == -1.
        # So we call dfs(root, -1).
        
        if dfs(root, -1):
            # Success
            # Return full mapping of length 20
            full_map = [-1] * 20
            for i in range(k):
                full_map[i] = mapping[i]
            return full_map
        else:
            return None

    # Main packing loop
    # We process trees from largest to smallest (19 down to 1)
    # Indices: 18 down to 0
    
    best_result = result[:]
    best_val = len(used_edges)
    
    # We might need to restart the packing of T19..T1 if we get stuck?
    # The backtracker for a single tree is exact (if it finds a solution, it returns it).
    # If it returns None, it means no embedding exists for that tree given the current used_edges.
    # Since we process largest first, if T19 fails, we can't pack it.
    # But maybe a different embedding of T20 (not identity) would allow T19?
    # The task says "position i is the image of vertex i of the fixed tree T_k".
    # It does NOT say T20 must be identity.
    # However, the previous code fixed T20 to identity.
    # If T20 is not identity, we have more freedom.
    # But T20 is a tree on 20 vertices. Embedding it into K20 is just a permutation.
    # Any permutation is valid for T20 as long as it's a bijection.
    # The edges used will be the edges of T20 mapped by that permutation.
    # Since T20 is a tree, its edge set is fixed up to isomorphism.
    # Wait, the "fixed tree T_k" implies the structure is fixed, but the embedding is what we choose.
    # So we can choose ANY embedding for T20.
    # To maximize flexibility for the rest, we should choose an embedding for T20 that leaves a "good" set of remaining edges for T19..T1.
    # However, trying all 20! permutations for T20 is impossible.
    # But we can just try a few random permutations for T20, and for each, try to pack the rest.
    
    # Let's modify the strategy:
    # 1. Try a few random embeddings for T20.
    # 2. For each, try to embed T19..T1 using the backtracker.
    # 3. Keep the best result.
    
    rng = random.Random(42) # Fixed seed for reproducibility of the search, or use time?
    # Let's use a random seed based on time for variety, but the task says seed=611 for trees.
    # The search randomness can be anything.
    
    best_overall = None
    best_score = 0
    
    num_restarts = 50 # Try 50 different T20 embeddings
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random permutation for T20
        perm20 = list(range(20))
        rng.shuffle(perm20)
        
        # Embed T20 with this permutation
        res = [None] * 20
        res[19] = perm20
        
        used = set()
        for u, v in trees[19]:
            gu = perm20[u]
            gv = perm20[v]
            if gu > gv: gu, gv = gv, gu
            used.add((gu, gv))
            
        current_score = 19 # Edges in T20
        
        # Try to embed T19..T1
        success_all = True
        for idx in range(18, -1, -1):
            k = idx + 1
            tree = trees[idx]
            
            # Use backtracker to embed this tree
            # We need a new RNG instance or just use the global one
            mapping = embed_tree(tree, k, used, rng)
            
            if mapping is None:
                # Failed to embed this tree
                # We can try to embed smaller trees?
                # The contract says "Entry k-1 is None or ...".
                # So we can leave it as None and try smaller ones?
                # Yes. "raw value is the total number of embedded tree edges".
                # So we should continue to try smaller trees even if a larger one fails.
                pass
            else:
                # Success
                res[idx] = mapping
                # Add edges to used
                for u, v in tree:
                    gu = mapping[u]
                    gv = mapping[v]
                    if gu > gv: gu, gv = gv, gu
                    used.add((gu, gv))
                    current_score += 1 # Each edge in tree adds 1 to score?
                    # Wait, "raw value is the total number of embedded tree edges".
                    # A tree with k vertices has k-1 edges.
                    # So if we embed T_k, we add k-1 to the score.
                    # My loop adds 1 per edge in the tree. Correct.
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_overall = res
            
        if best_score == 190:
            break
            
    if best_overall is None:
        # Fallback: just T20 identity
        best_overall = [None] * 20
        best_overall[19] = list(range(20))
        best_score = 19
        
    return best_overall

