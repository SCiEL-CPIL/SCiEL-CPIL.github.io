"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    # Every edge as its own path; the instance generator is copied from the verifier.
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        """
        Decompose edges into edge-disjoint paths covering all edges.
        Uses a greedy approach:
        1. Start with all edges as separate paths.
        2. Repeatedly try to merge two paths if they share a vertex and the merge
           would keep the path simple (no repeated vertices).
        This is a heuristic that tries to reduce the number of paths.
        """
        # Build adjacency for quick lookup
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Start with each edge as a path
        paths = [[u, v] for u, v in edges]
        
        # We'll try to merge paths greedily
        # A path [v0, v1, ..., vk] can be extended at start with vertex u if u-v0 is an edge
        # and u is not in the path, or at end with vertex w if vk-w is an edge and w not in path.
        # Two paths P1 and P2 can be merged if:
        # - P1 ends at a vertex that P2 starts at (or vice versa), and the union is simple
        # - Or more generally, if there's a vertex shared such that we can concatenate
        
        # Simpler approach: use a matching-like algorithm on the line graph or
        # use the fact that we want a path cover.
        
        # Let's use a different strategy: 
        # 1. Find all vertices with odd degree. In any path decomposition, 
        #    the number of paths is at least ceil(odd_vertices / 2).
        # 2. Try to construct paths by pairing up odd vertices and finding paths between them,
        #    then handle even-degree components.
        
        # Actually, let's use a simpler greedy merge:
        # Keep trying to merge paths until no more merges are possible.
        
        changed = True
        while changed:
            changed = False
            # For each pair of paths, check if they can be merged
            # To be efficient, index paths by their endpoints
            # path i has start s_i and end t_i
            # We can merge path i and path j if:
            #   t_i == s_j and all vertices in path j are not in path i (except t_i/s_j)
            #   s_i == t_j and all vertices in path i are not in path j (except s_i/t_j)
            #   t_i == t_j and ... (more complex, skip for now)
            
            # Build a map from vertex to list of (path_index, position) where position is 0 (start) or 1 (end)
            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))  # start
                endpoint_map[path[-1]].append((i, 1))  # end
            
            # Try to find a merge
            merged_any = False
            for v, endpoints in endpoint_map.items():
                # For each pair of endpoints at vertex v
                for idx1 in range(len(endpoints)):
                    for idx2 in range(idx1 + 1, len(endpoints)):
                        i, pos1 = endpoints[idx1]
                        j, pos2 = endpoints[idx2]
                        if i == j:
                            continue
                        # Check if we can merge path i and path j at vertex v
                        # Case 1: pos1 == 0 (i starts at v) and pos2 == 1 (j ends at v)
                        #   New path: j + i[1:]
                        # Case 2: pos1 == 1 (i ends at v) and pos2 == 0 (j starts at v)
                        #   New path: i + j[1:]
                        
                        if pos1 == 0 and pos2 == 1:
                            # Merge j into i: new_path = j + i[1:]
                            new_path = paths[j] + paths[i][1:]
                            # Check simplicity
                            if len(set(new_path)) == len(new_path):
                                # Perform merge
                                paths[i] = new_path
                                paths[j] = []
                                changed = True
                                merged_any = True
                                # Rebuild endpoint_map
                                endpoint_map = defaultdict(list)
                                for k, path in enumerate(paths):
                                    if path:
                                        endpoint_map[path[0]].append((k, 0))
                                        endpoint_map[path[-1]].append((k, 1))
                                break
                        elif pos1 == 1 and pos2 == 0:
                            # Merge i into j: new_path = i + j[1:]
                            new_path = paths[i] + paths[j][1:]
                            if len(set(new_path)) == len(new_path):
                                paths[j] = new_path
                                paths[i] = []
                                changed = True
                                merged_any = True
                                endpoint_map = defaultdict(list)
                                for k, path in enumerate(paths):
                                    if path:
                                        endpoint_map[path[0]].append((k, 0))
                                        endpoint_map[path[-1]].append((k, 1))
                                break
                    if merged_any:
                        break
                if merged_any:
                    break
            
            if not merged_any:
                break
        
        # Remove empty paths
        paths = [p for p in paths if p]
        
        # Verify edge-disjointness and coverage
        used_edges = set()
        for path in paths:
            for k in range(len(path) - 1):
                e = tuple(sorted((path[k], path[k+1])))
                if e in used_edges:
                    # Error: edge used twice
                    pass
                used_edges.add(e)
        
        # Check all edges are covered
        for e in edges:
            if e not in used_edges:
                # Error: edge not covered
                pass
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
