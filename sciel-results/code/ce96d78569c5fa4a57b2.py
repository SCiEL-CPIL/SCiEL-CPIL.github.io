"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import networkx as nx
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        graph = nx.Graph()
        for u, v in edges:
            graph.add_edge(u, v)
        
        # Initialize paths as single edges
        paths = [[u, v] for u, v in edges]
        
        # Helper to check if two paths can be merged
        # P1 and P2 can be merged if they share exactly one vertex v,
        # and that vertex is an endpoint of both.
        # And the resulting path is simple (which is guaranteed if they are vertex-disjoint except for v).
        
        def get_endpoints(path):
            return path[0], path[-1]
            
        def can_merge(p1, p2):
            # Check if they share exactly one vertex
            set1 = set(p1)
            set2 = set(p2)
            common = set1 & set2
            if len(common) != 1:
                return False
            v = common.pop()
            
            # v must be an endpoint of p1
            if v not in (p1[0], p1[-1]):
                return False
            # v must be an endpoint of p2
            if v not in (p2[0], p2[-1]):
                return False
            
            return True

        def merge_paths(p1, p2):
            # Determine orientation
            # p1 ends at v, p2 starts at v -> p1 + p2[1:]
            # p1 ends at v, p2 ends at v -> p1 + reversed(p2)[1:]
            # etc.
            
            v1_start, v1_end = p1[0], p1[-1]
            v2_start, v2_end = p2[0], p2[-1]
            
            # Find the common vertex v
            common = set(p1) & set(p2)
            v = common.pop()
            
            # Case 1: p1 ends at v, p2 starts at v
            if v1_end == v and v2_start == v:
                return p1 + p2[1:]
            # Case 2: p1 starts at v, p2 ends at v
            elif v1_start == v and v2_end == v:
                return p2 + p1[1:]
            # Case 3: p1 ends at v, p2 ends at v
            elif v1_end == v and v2_end == v:
                return p1 + p2[::-1][1:]
            # Case 4: p1 starts at v, p2 starts at v
            else: # v1_start == v and v2_start == v
                return p2[::-1] + p1[1:]

        # Iterative matching
        changed = True
        while changed:
            changed = False
            n_paths = len(paths)
            if n_paths < 2:
                break
            
            # Build auxiliary graph for matching
            # Nodes are path indices 0..n_paths-1
            # Edge (i, j) if paths[i] and paths[j] can merge
            merge_graph = nx.Graph()
            merge_graph.add_nodes_from(range(n_paths))
            
            # Optimization: Only check pairs that share a vertex
            # Group paths by their endpoints?
            # Or just O(N^2) check since N is small (initially ~100, reduces quickly)
            
            for i in range(n_paths):
                for j in range(i + 1, n_paths):
                    if can_merge(paths[i], paths[j]):
                        merge_graph.add_edge(i, j)
                        
            # Find maximum matching
            # NetworkX has max_weight_matching, but for cardinality we can use it with weight 1
            # Or use a simple greedy if N is small?
            # Let's use networkx.algorithms.matching.max_weight_matching
            matching = nx.max_weight_matching(merge_graph, maxcardinality=True)
            
            if not matching:
                break
                
            # Apply merges
            # We need to be careful: if i is matched with j, we merge them and remove i, j.
            # The new path index will be the minimum of i, j? Or we just rebuild the list.
            
            new_paths = []
            used = set()
            for i, j in matching:
                if i in used or j in used:
                    continue
                merged = merge_paths(paths[i], paths[j])
                new_paths.append(merged)
                used.add(i)
                used.add(j)
                
            for k in range(n_paths):
                if k not in used:
                    new_paths.append(paths[k])
                    
            paths = new_paths
            changed = True

        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
