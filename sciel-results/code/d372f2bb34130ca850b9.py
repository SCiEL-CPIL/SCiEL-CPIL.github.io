"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

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
    
    used_edges = set()
    result = [None] * n
    
    # Embed from largest to smallest (k=20 down to 1)
    # This is a structural change: larger trees are more constrained and should be placed first.
    order = list(range(n - 1, -1, -1))
    
    for idx in order:
        if time.time() > deadline:
            break
        
        k = idx + 1
        tree_edges = trees[idx]
        
        if k == 1:
            # T1: single vertex. Any vertex works.
            # Since it uses no edges, it doesn't affect used_edges.
            # We just need to return a list of 1 distinct int.
            result[idx] = [0]
            continue
        
        if k == 2:
            # T2: single edge. Find any unused edge.
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add((u, v))
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen if we have enough edges, but handle gracefully
                # If we can't find an edge, we leave it as None, which is invalid for the whole result.
                # But since total edges match, this implies a previous step failed or logic error.
                pass
            continue
        
        # For k >= 3, use backtracking with random restarts
        embedding = _embed_tree_largest_first(tree_edges, used_edges, n, deadline)
        
        if embedding is not None:
            result[idx] = embedding
            for (a, b) in tree_edges:
                u, v = embedding[a], embedding[b]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        else:
            # If backtracking fails, we cannot recover easily without restarting the whole process
            # or trying different orders. Given the time constraint, we return what we have.
            # The problem states "Any broken rule makes the whole result invalid".
            # So if we fail to embed one, the score is 0.
            # To maximize chance, we rely on the robustness of the backtracker.
            pass

    # Ensure types are correct
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            # Ensure list of ints
            final_result.append([int(x) for x in result[i]])
            
    return final_result

def _embed_tree_largest_first(tree_edges, used_edges, n, deadline):
    """
    Embeds a tree into K_n avoiding used_edges.
    Uses backtracking with forward checking and heuristics:
    1. Map high-degree tree vertices first.
    2. Map tree vertices with many mapped neighbors first (fail-first).
    3. Forward checking: verify unassigned vertices still have valid hosts.
    4. Randomize host vertex selection order to escape local minima.
    """
    k = len(tree_edges) + 1
    
    # Build adjacency list for the tree
    tree_adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
        
    # Pre-calculate degrees
    tree_deg = [len(tree_adj[i]) for i in range(k)]
    
    # We will try multiple restarts with different random seeds for host ordering
    # to increase the chance of finding a valid embedding.
    max_restarts = 200
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
            
        # State
        mapping = [-1] * k
        used_host_verts = [False] * n
        
        # Determine the order in which we assign tree vertices.
        # Heuristic: 
        # 1. Start with a vertex of max degree.
        # 2. Next, pick an unassigned vertex that has the most neighbors already assigned.
        #    This forces constraints early.
        
        assigned_order = []
        assigned_set = set()
        
        # Pick start vertex: max degree
        start_v = max(range(k), key=lambda x: tree_deg[x])
        assigned_order.append(start_v)
        assigned_set.add(start_v)
        
        # Dynamic selection of next vertex
        while len(assigned_order) < k:
            best_v = -1
            max_mapped_neighbors = -1
            for v in range(k):
                if v in assigned_set:
                    continue
                count = sum(1 for nb in tree_adj[v] if nb in assigned_set)
                if count > max_mapped_neighbors:
                    max_mapped_neighbors = count
                    best_v = v
            if best_v == -1:
                # Should not happen for a connected tree
                break
            assigned_order.append(best_v)
            assigned_set.add(best_v)
            
        # Host vertex order: randomized
        host_order = list(range(n))
        rng.shuffle(host_order)
        
        # Precompute: for each tree vertex, which host vertices are valid given used_edges
        # This is dynamic, so we'll check on the fly
        
        # Backtracking function with forward checking
        # idx: index in assigned_order
        def backtrack(idx):
            if time.time() > deadline:
                return False
                
            if idx == k:
                return True
                
            tv = assigned_order[idx]
            
            # For each candidate host vertex
            for hv in host_order:
                if used_host_verts[hv]:
                    continue
                
                # Check validity:
                # 1. Distinctness is handled by used_host_verts
                # 2. Edge disjointness: for all neighbors 'nb' of tv that are ALREADY MAPPED,
                #    the edge (hv, mapping[nb]) must not be in used_edges.
                
                valid = True
                for nb in tree_adj[tv]:
                    if mapping[nb] != -1:
                        nhv = mapping[nb]
                        u, v = hv, nhv
                        if u > v:
                            u, v = v, u
                        if (u, v) in used_edges:
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # Forward checking: for each unassigned tree vertex, check if it still has
                # at least one valid host vertex
                feasible = True
                for j in range(idx + 1, k):
                    tv2 = assigned_order[j]
                    has_valid_host = False
                    for hv2 in host_order:
                        if used_host_verts[hv2]:
                            continue
                        # Check edge disjointness for tv2 with hv2
                        ok = True
                        for nb2 in tree_adj[tv2]:
                            if mapping[nb2] != -1:
                                nhv2 = mapping[nb2]
                                u2, v2 = hv2, nhv2
                                if u2 > v2:
                                    u2, v2 = v2, u2
                                if (u2, v2) in used_edges:
                                    ok = False
                                    break
                        if ok:
                            has_valid_host = True
                            break
                    if not has_valid_host:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                    
                mapping[tv] = hv
                used_host_verts[hv] = True
                
                if backtrack(idx + 1):
                    return True
                    
                mapping[tv] = -1
                used_host_verts[hv] = False
                    
        if backtrack(0):
            return mapping
            
    return None

# EVOLVE-BLOCK-END
