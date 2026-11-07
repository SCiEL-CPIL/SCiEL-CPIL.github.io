"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    
    # Edge set for K20
    K20_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            K20_edges.add((u, v))
            
    used_edges = set()
    embeddings = [None] * 20
    
    # T1 is empty
    embeddings[0] = []
    
    # T2 is a single edge
    embeddings[1] = [0, 1]
    used_edges.add((0, 1))
    
    # For T3..T20, try to embed greedily or with small backtracking
    # We process in order of increasing k (3 to 20)
    # For each tree, we try to find a mapping from tree vertices 0..k-1 to host vertices 0..19
    # such that all tree edges map to distinct unused K20 edges.
    
    # To improve score, we want to embed as many trees as possible.
    # The raw value is the sum of edges of embedded trees.
    # T_k has k-1 edges.
    # Max total edges is sum_{k=1}^{20} (k-1) = 190.
    
    # A simple heuristic: for each tree, try random permutations of host vertices?
    # That's 20! which is too big.
    # Instead, we can use a greedy approach with backtracking limited in depth, or just random restarts.
    
    # Let's try a simple greedy algorithm with randomization and restarts.
    # For each tree T_k, we build an embedding vertex by vertex.
    
    # Since we have 150s, we can afford some search.
    # Let's try to embed trees in order of difficulty? Smaller trees are easier.
    # Actually, T20 is a tree on 20 vertices. It must use a perfect matching of K20? No, just a spanning tree.
    # If we embed T20 using all 20 vertices, it uses 19 edges.
    # The remaining 190 - 19 = 171 edges are available for T1..T19.
    # T1..T19 have 1+2+...+18 = 171 edges total.
    # So it is theoretically possible to embed ALL trees if we can pack them perfectly.
    # This is a "tree packing" problem.
    
    # Strategy:
    # 1. Embed T20 first? Or last?
    #    If we embed T20 last, we have the most flexibility for the smaller trees, and T20 just needs to fit in the remaining graph.
    #    The remaining graph after embedding T1..T19 will have 190 - 171 = 19 edges.
    #    T20 needs 19 edges. So the remaining graph MUST be exactly a spanning tree of K20.
    #    This is a very tight constraint. It implies that T1..T19 must partition the edges of K20 \ T20.
    #    This is unlikely to happen by chance with greedy methods.
    
    # Alternative: Randomized greedy with backtracking.
    # Let's try to embed trees one by one.
    # We'll use a simple recursive backtracker for each tree, with a node limit.
    
    def try_embed_tree(tree_edges, k, used_edges, deadline):
        # tree_edges: list of (u, v)
        # k: number of vertices
        # Returns: list of host vertices for tree vertices 0..k-1, or None
        
        # We need to map tree vertices to host vertices.
        # Let's fix vertex 0 of the tree to host vertex 0 (by symmetry of K20).
        # Then map the rest.
        
        # Actually, K20 is vertex-transitive, so we can fix tree vertex 0 to any host vertex.
        # Let's fix tree vertex 0 to host 0.
        
        # We need to assign hosts to tree vertices 1..k-1.
        # Let's use a simple backtracking search.
        
        # Order of assignment: BFS order of the tree starting from 0.
        # This ensures that when we assign a vertex, its parent is already assigned.
        
        # Build adjacency list for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS order
        visited = [False] * k
        order = [0]
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    queue.append(v)
                    
        # Now we assign hosts to vertices in `order`.
        # host[0] = 0
        host = [None] * k
        host[0] = 0
        
        # We'll use a recursive function.
        # To avoid deep recursion and overhead, we might limit the search.
        # But k is at most 20.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                # Check all edges are valid and unused
                for u, v in tree_edges:
                    hu, hv = host[u], host[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        return False
                # If we get here, it's a valid embedding
                return True
            
            v = order[idx]
            # Try all available host vertices
            # Heuristic: try vertices connected to already assigned neighbors?
            # Or just random order?
            # Let's try random order of host vertices 0..19.
            candidates = list(range(20))
            Random(1234 + idx).shuffle(candidates) # Fixed seed for reproducibility in search? No, let's use global rng
            
            for h in candidates:
                if h in host[:idx]: # Already used?
                    # Wait, host vertices must be distinct for the tree embedding?
                    # "list of k distinct int host vertices" -> Yes, distinct.
                    continue
                    
                # Check if edge to parent is valid
                # Find parent of v in the tree (the neighbor in order before it? No, the one that discovered it)
                # We need to know the parent in the BFS tree.
                # Let's store parent in BFS.
                
                # Actually, we just need to check that the edge (host[v], h) is not in used_edges.
                # But we don't know host[v] yet? No, h is the candidate for v.
                # We need to check edges from v to its already assigned neighbors.
                
                valid = True
                for u in adj[v]:
                    if host[u] is not None:
                        edge = (min(host[u], h), max(host[u], h))
                        if edge in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                    
                host[v] = h
                if backtrack(idx + 1):
                    return True
                host[v] = None
                
            return False
            
        # We need parent info for efficient checking?
        # The above checks all neighbors. Since tree is sparse, it's fine.
        
        # Optimization: Check edge availability before recursion?
        # The above does it.
        
        # Let's add a time check inside the loop too?
        # The check at the start of backtrack is good.
        
        return backtrack(1) if k > 1 else True
        
    # We need to store the final host mapping if backtrack returns True.
    # The `host` list is modified in place.
    
    # Let's refine try_embed_tree to return the mapping.
    
    def solve_tree(tree_edges, k, used_edges, deadline):
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        visited = [False] * k
        order = [0]
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    queue.append(v)
                    
        host = [None] * k
        host[0] = 0
        
        # Use a set for used hosts for O(1) lookup
        used_hosts = {0}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                # Final check for all edges (redundant if we checked incrementally, but safe)
                for u, v in tree_edges:
                    hu, hv = host[u], host[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        return False
                return True
            
            v = order[idx]
            # Candidates: 0..19 not in used_hosts
            # Sort candidates? Maybe by degree in remaining graph?
            # For now, random.
            candidates = [h for h in range(20) if h not in used_hosts]
            # Shuffle with a seed based on idx and tree index?
            # Let's just use a global RNG.
            rng.shuffle(candidates)
            
            for h in candidates:
                valid = True
                for u in adj[v]:
                    if host[u] is not None:
                        edge = (min(host[u], h), max(host[u], h))
                        if edge in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                    
                host[v] = h
                used_hosts.add(h)
                if backtrack(idx + 1):
                    return True
                host[v] = None
                used_hosts.remove(h)
                
            return False
            
        if k == 1:
            return [0]
            
        if backtrack(1):
            return host
        return None

    rng = Random(42)
    
    # Order of embedding:
    # Let's try embedding from largest to smallest?
    # T20 is the hardest. If we fail T20, we lose 19 edges.
    # If we embed small ones first, we might block T20.
    # If we embed T20 first, we use 19 edges. Then we have 171 edges left for 171 edges of T1..T19.
    # This is a perfect packing if successful.
    
    # Let's try embedding T20 first.
    # Then T19, T18, ...
    
    # But wait, if we embed T20 first, the remaining graph is K20 minus a spanning tree.
    # This graph is 18-regular? No, K20 is 19-regular. Minus a tree (degrees vary).
    # The remaining graph is highly connected.
    
    # Let's try this order: 20, 19, 18, ..., 1.
    
    order_of_trees = list(range(20, 0, -1))
    
    # T1 is trivial
    # T2 is trivial
    
    # We'll keep track of used_edges.
    
    # Restart loop?
    # If we fail to embed a tree, we can restart the whole process with a different random seed/order?
    # Given the tight constraint, a single pass might fail.
    
    # Let's do a few restarts.
    
    best_score = 0
    best_embeddings = [None] * 20
    
    for restart in range(100): # Up to 100 restarts
        if time.time() > deadline - 5: break
        
        used_edges = set()
        embeddings = [None] * 20
        success = True
        
        # Embed T20
        if time.time() > deadline - 5: break
        res = solve_tree(trees[19], 20, used_edges, deadline)
        if res is None:
            # Try a different randomization?
            # Maybe skip T20 for now?
            # If we skip T20, we can still embed others.
            # Let's try to embed others and see if we can fit T20 later?
            # No, let's just mark it failed and continue with others?
            # But the contract says "Entry k-1 is None or a list".
            # So we can leave it None.
            # But we want to maximize score.
            # If T20 fails, we lose 19 points.
            # Let's try to embed T1..T19 and see if T20 fits in the hole?
            # That's complex.
            
            # Let's just try the next restart with a different RNG seed.
            continue
            
        embeddings[19] = res
        for u, v in trees[19]:
            used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
            
        # Embed T19..T1
        for k in range(19, 1, -1): # 19 down to 2
            if time.time() > deadline - 5: break
            res = solve_tree(trees[k-1], k, used_edges, deadline)
            if res is None:
                success = False
                break
            embeddings[k-1] = res
            for u, v in trees[k-1]:
                used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
                
        if success:
            embeddings[0] = [] # T1
            # Calculate score
            score = 0
            for k in range(1, 21):
                if embeddings[k-1] is not None:
                    score += k - 1
            if score > best_score:
                best_score = score
                best_embeddings = embeddings[:]
                
            if best_score == 190:
                break # Perfect packing found
                
    # If best_score is 0, try a simpler fallback
    if best_score == 0:
        # Just embed T20
        res = solve_tree(trees[19], 20, set(), deadline)
        if res:
            best_embeddings[19] = res
            best_score = 19
        else:
            # Identity map
            best_embeddings[19] = list(range(20))
            best_score = 19
            
    return best_embeddings

# EVOLVE-BLOCK-END
