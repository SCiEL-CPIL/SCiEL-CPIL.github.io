"""Packing trees T1..T20 into K20 using backtracking search."""
import time
import sys
from random import Random
from collections import defaultdict

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
    deadline = time.time() + 140  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    
    # Edge usage tracking: set of (min(u,v), max(u,v)) for used edges
    used_edges = set()
    
    # Result: list of 20 entries, entry k-1 is None or list of k ints
    result = [None] * 20
    
    # Trees in order of embedding (largest first)
    tree_order = list(range(19, -1, -1))  # indices 19 down to 0, i.e., T20 first
    
    def embed_tree(tree_edges, used_edges_set, time_limit=10.0):
        """Find an embedding of the tree into K20 using unused edges with efficient backtracking."""
        if not tree_edges:
            # T1: return [0] (any single vertex)
            return [0]
        
        verts = set()
        for a, b in tree_edges:
            verts.add(a)
            verts.add(b)
        k = len(verts)
        
        if k == 2:
            # T2: two vertices, one edge
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges_set:
                        return [u, v]
            return None
        
        # Build adjacency
        adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Compute degree for each tree vertex
        degree = [len(adj[v]) for v in range(k)]
        
        # Order vertices by decreasing degree (most constrained first)
        # This helps backtracking prune earlier
        order = sorted(range(k), key=lambda v: -degree[v])
        
        # Track which host vertices are used
        host_used = [False] * 20
        assignment = [-1] * k  # assignment[i] = host vertex for tree vertex i, or -1 if unassigned
        
        # Precompute available edges for each pair of host vertices
        # available[u][v] = True if edge (u,v) is not in used_edges_set
        # We'll use a set for O(1) lookup
        
        def backtrack(pos):
            """Try to assign tree vertex order[pos] to a host vertex."""
            if time.time() > time_limit:
                return False
            
            if pos == k:
                # All vertices assigned - verify all edges are valid
                for a, b in tree_edges:
                    ha = assignment[a]
                    hb = assignment[b]
                    e1 = min(ha, hb)
                    e2 = max(ha, hb)
                    if (e1, e2) in used_edges_set:
                        return False
                return True
            
            tree_v = order[pos]
            
            # For each candidate host vertex, check if it's valid
            for hv in range(20):
                if host_used[hv]:
                    continue
                
                # Check edges to already-assigned neighbors
                valid = True
                for nb in adj[tree_v]:
                    if assignment[nb] != -1:
                        e1 = min(hv, assignment[nb])
                        e2 = max(hv, assignment[nb])
                        if (e1, e2) in used_edges_set:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Additional forward checking: ensure unassigned neighbors still have options
                # This is expensive, so only do it for high-degree vertices
                if degree[tree_v] > 3:
                    feasible = True
                    for nb in adj[tree_v]:
                        if assignment[nb] == -1:
                            # Check if nb has at least one available host vertex
                            has_option = False
                            for hv_nb in range(20):
                                if host_used[hv_nb]:
                                    continue
                                # Check edge from hv to hv_nb
                                e1 = min(hv, hv_nb)
                                e2 = max(hv, hv_nb)
                                if (e1, e2) not in used_edges_set:
                                    has_option = True
                                    break
                            if not has_option:
                                feasible = False
                                break
                    if not feasible:
                        continue
                
                # Assign and recurse
                assignment[tree_v] = hv
                host_used[hv] = True
                if backtrack(pos + 1):
                    return True
                host_used[hv] = False
                assignment[tree_v] = -1
            
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Now do the packing: embed trees in order from largest to smallest
    for idx in tree_order:
        if time.time() > deadline:
            break
        tree_edges = trees[idx]
        
        # Time limit per tree: use more time for larger trees
        time_limit = min(10.0, (deadline - time.time()) * 0.8)
        if time_limit <= 0:
            break
            
        emb = embed_tree(tree_edges, used_edges, time_limit)
        if emb is None:
            result[idx] = None
        else:
            result[idx] = emb
            # Update used edges
            for a, b in tree_edges:
                ha = emb[a]
                hb = emb[b]
                e1 = min(ha, hb)
                e2 = max(ha, hb)
                used_edges.add((e1, e2))
    
    # Verify the result
    for i in range(20):
        if result[i] is not None:
            if len(result[i]) != i + 1:
                result[i] = None
            elif len(set(result[i])) != i + 1:
                result[i] = None
            elif any(not isinstance(x, int) or x < 0 or x > 19 for x in result[i]):
                result[i] = None
    
    # Verify edge-disjointness
    used_edges_check = set()
    for i in range(20):
        if result[i] is None:
            continue
        tree_edges = trees[i]
        for a, b in tree_edges:
            ha = result[i][a]
            hb = result[i][b]
            e1 = min(ha, hb)
            e2 = max(ha, hb)
            if (e1, e2) in used_edges_check:
                result[i] = None
                break
            used_edges_check.add((e1, e2))
    
    return result
