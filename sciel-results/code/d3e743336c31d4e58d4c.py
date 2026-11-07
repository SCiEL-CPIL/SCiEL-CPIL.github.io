# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

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
    deadline = time.time() + 115.0
    n = 20
    trees = instance_trees(n, 611)
    
    # Track used edges as a set of frozensets for O(1) lookup
    used_edges = set()
    result = [None] * n
    
    # Embed trees from largest to smallest
    order = list(range(n - 1, -1, -1))
    
    for idx in order:
        if time.time() > deadline:
            break
        k = idx + 1
        tree_edges = trees[idx]
        
        if k == 1:
            result[idx] = [0]
            continue
        
        if k == 2:
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if frozenset((u, v)) not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add(frozenset((u, v)))
                        found = True
                        break
                if found:
                    break
            continue
        
        # Build tree adjacency
        tree_adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            tree_adj[a].append(b)
            tree_adj[b].append(a)
        
        # Find a good starting vertex: highest degree
        start_v = max(range(k), key=lambda v: len(tree_adj[v]))
        
        # Pre-compute: for each tree vertex, its neighbors
        # We'll use a recursive backtracking with MRV heuristic
        
        def try_embed(remaining_deadline):
            # State: mapping tree_vertex -> host_vertex
            # We'll use a list for the current assignment
            embedding = [-1] * k
            
            # Order tree vertices for assignment: start with start_v, then use MRV
            # For simplicity, we'll use a fixed order but prioritize high-degree vertices
            # Actually, let's use a dynamic MRV approach
            
            # First, assign start_v to each possible host vertex
            for h in range(n):
                if time.time() > remaining_deadline:
                    return None
                embedding[start_v] = h
                
                if _backtrack_mrv(embedding, tree_adj, used_edges, k, n, remaining_deadline):
                    return embedding[:]
                embedding[start_v] = -1
            
            return None
        
        def _backtrack_mrv(embedding, tree_adj, used_edges, k, n, deadline):
            if time.time() > deadline:
                return False
            
            # Check if all vertices are assigned
            unassigned = [v for v in range(k) if embedding[v] == -1]
            if not unassigned:
                # Verify all edges are valid
                for a, b in tree_edges:
                    ua, ub = embedding[a], embedding[b]
                    if frozenset((ua, ub)) in used_edges:
                        return False
                return True
            
            # MRV: pick the unassigned vertex with the fewest valid host options
            best_v = -1
            best_candidates = None
            min_count = n + 1
            
            for v in unassigned:
                # Count valid host vertices for v
                valid_hosts = []
                for h in range(n):
                    # Check injectivity within this tree
                    if h in embedding:
                        continue
                    # Check edge constraints with assigned neighbors
                    valid = True
                    for u in tree_adj[v]:
                        if embedding[u] != -1:
                            if frozenset((embedding[u], h)) in used_edges:
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)
                
                if len(valid_hosts) < min_count:
                    min_count = len(valid_hosts)
                    best_v = v
                    best_candidates = valid_hosts
                    if min_count == 0:
                        return False
            
            if best_v == -1:
                return False
            
            # LCV: order candidates by how many options they leave for other unassigned vertices
            # For simplicity, we'll just shuffle if we want randomization
            # But let's try a simple heuristic: prefer hosts that are "central"
            random.shuffle(best_candidates)
            
            for h in best_candidates:
                embedding[best_v] = h
                if _backtrack_mrv(embedding, tree_adj, used_edges, k, n, deadline):
                    return True
                embedding[best_v] = -1
            
            return False
        
        embedding = try_embed(deadline)
        if embedding is not None:
            result[idx] = embedding
            for a, b in tree_edges:
                ua, ub = embedding[a], embedding[b]
                used_edges.add(frozenset((ua, ub)))
    
    # Convert to proper format
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append(list(result[i]))
    
    return final_result
# EVOLVE-BLOCK-END
