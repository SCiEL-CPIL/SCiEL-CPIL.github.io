"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

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
    deadline = start_time + 140.0
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Embed trees from smallest to largest
    # T1: single vertex, no edges
    # T2: single edge
    # T3..T20: larger trees
    
    used_edges = set()
    result = [None] * 20
    
    # T1: no edges, map to vertex 0
    result[0] = [0]
    
    # T2: single edge
    # Try to find an available edge
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in used_edges:
                result[1] = [u, v]
                used_edges.add((u, v))
                break
        if result[1] is not None:
            break
    
    # For T3..T20, use a greedy embedding with backtracking
    # Process in order of increasing size for better success rate
    for k in range(3, 21):
        if time.time() > deadline:
            break
        
        tree = trees[k-1]
        adj = [[] for _ in range(k)]
        for a, b in tree:
            adj[a].append(b)
            adj[b].append(a)
        
        # Find a leaf to start
        root = 0
        for v in range(k):
            if len(adj[v]) == 1:
                root = v
                break
        
        # Try to embed using DFS with backtracking
        def embed_dfs(t_node, h_node, mapping, visited_t, visited_e):
            if time.time() > deadline:
                return None
            
            mapping[t_node] = h_node
            visited_t.add(t_node)
            
            neighbors = [nb for nb in adj[t_node] if nb not in visited_t]
            
            def assign(neigh_idx, local_map, local_vt, local_ve):
                if time.time() > deadline:
                    return None
                
                if neigh_idx == len(neighbors):
                    for t_nb in neighbors:
                        if t_nb in local_map:
                            h_nb = local_map[t_nb]
                            sub = embed_dfs(t_nb, h_nb, local_map, local_vt, local_ve)
                            if sub is None:
                                return None
                    return local_map
                
                t_nb = neighbors[neigh_idx]
                if t_nb in local_map:
                    return assign(neigh_idx + 1, local_map, local_vt, local_ve)
                
                for h_v in range(n):
                    if h_v == h_node:
                        continue
                    if h_v in local_map.values():
                        continue
                    edge = (min(h_node, h_v), max(h_node, h_v))
                    if edge in local_ve or edge in used_edges:
                        continue
                    
                    new_map = dict(local_map)
                    new_map[t_nb] = h_v
                    new_ve = set(local_ve)
                    new_ve.add(edge)
                    
                    res = assign(neigh_idx + 1, new_map, local_vt, new_ve)
                    if res is not None:
                        return res
                return None
            
            return assign(0, mapping, visited_t, visited_e)
        
        success = False
        for host_root in range(n):
            if time.time() > deadline:
                break
            mapping = {}
            visited_t = set()
            visited_e = set()
            res = embed_dfs(root, host_root, mapping, visited_t, visited_e)
            if res is not None:
                # Verify
                valid = True
                for a, b in tree:
                    ha, hb = res[a], res[b]
                    edge = (min(ha, hb), max(ha, hb))
                    if edge in used_edges:
                        valid = False
                        break
                if valid:
                    result[k-1] = [res[i] for i in range(k)]
                    for a, b in tree:
                        ha, hb = res[a], res[b]
                        used_edges.add((min(ha, hb), max(ha, hb)))
                    success = True
                    break
        
        if not success:
            # Fallback: skip this tree
            result[k-1] = None
    
    # Ensure all values are proper ints
    final = []
    for i in range(20):
        if result[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in result[i]])
    
    return final
# EVOLVE-BLOCK-END
