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
        # Initialize paths as single edges
        # Store paths as lists of vertices
        paths = [[u, v] for u, v in edges]
        
        # Helper to check if two paths can be merged
        def can_merge(p1, p2):
            # Check if they share exactly one vertex
            set1 = set(p1)
            set2 = set(p2)
            common = set1 & set2
            if len(common) != 1:
                return None
            
            v = common.pop()
            
            # v must be an endpoint of p1
            if v == p1[0]:
                idx1 = 0
            elif v == p1[-1]:
                idx1 = 1
            else:
                return None
                
            # v must be an endpoint of p2
            if v == p2[0]:
                idx2 = 0
            elif v == p2[-1]:
                idx2 = 1
            else:
                return None
            
            return (idx1, idx2)

        def merge_paths(p1, p2, idx1, idx2):
            # idx1 is 0 if p1[0] is the shared vertex, 1 if p1[-1] is
            # idx2 is 0 if p2[0] is the shared vertex, 1 if p2[-1] is
            
            # We want to concatenate them such that the shared vertex is in the middle
            # Case 1: p1 ends at v (idx1=1), p2 starts at v (idx2=0)
            #   p1 + p2[1:]
            # Case 2: p1 starts at v (idx1=0), p2 ends at v (idx2=1)
            #   p2 + p1[1:]
            # Case 3: p1 ends at v (idx1=1), p2 ends at v (idx2=1)
            #   p1 + reversed(p2)[1:]
            # Case 4: p1 starts at v (idx1=0), p2 starts at v (idx2=0)
            #   reversed(p2) + p1[1:]
            
            if idx1 == 1 and idx2 == 0:
                return p1 + p2[1:]
            elif idx1 == 0 and idx2 == 1:
                return p2 + p1[1:]
            elif idx1 == 1 and idx2 == 1:
                return p1 + p2[::-1][1:]
            else: # idx1 == 0 and idx2 == 0
                return p2[::-1] + p1[1:]

        # Iterative matching
        changed = True
        while changed:
            changed = False
            n_paths = len(paths)
            if n_paths < 2:
                break
            
            # Build auxiliary graph for matching
            # Nodes are (path_index, end_index) where end_index is 0 (start) or 1 (end)
            merge_graph = nx.Graph()
            # Add all endpoint nodes
            for i in range(n_paths):
                merge_graph.add_node((i, 0))
                merge_graph.add_node((i, 1))
            
            # Find all possible merges
            for i in range(n_paths):
                for j in range(i + 1, n_paths):
                    res = can_merge(paths[i], paths[j])
                    if res:
                        idx1, idx2 = res
                        # Connect the specific endpoints involved in the merge
                        merge_graph.add_edge((i, idx1), (j, idx2))
                        
            # Find maximum matching
            # max_weight_matching with maxcardinality=True finds a maximum cardinality matching
            matching = nx.max_weight_matching(merge_graph, maxcardinality=True)
            
            if not matching:
                break
                
            # Apply merges
            # The matching is a set of edges in merge_graph.
            # Each edge ((i, e1), (j, e2)) means merge paths i and j using endpoints e1 and e2.
            # Since it's a matching, each path index i appears in at most one edge (as i,0 or i,1).
            # Actually, a path has two endpoints. Could both be matched?
            # If path i is matched via endpoint 0 with path k, and via endpoint 1 with path l?
            # That would mean path i is merged with k and l simultaneously.
            # But path i is a single entity. It can only be part of one merge per iteration.
            # The matching is on the *endpoint nodes*.
            # Node (i, 0) and (i, 1) are distinct nodes in merge_graph.
            # So it is possible for (i, 0) to be matched with (j, 0) and (i, 1) to be matched with (k, 1) in the same matching.
            # This would imply merging path i with path j AND path i with path k.
            # This is invalid because path i cannot be split.
            # However, `max_weight_matching` on the auxiliary graph treats (i,0) and (i,1) as independent nodes.
            # We need to ensure that for each path i, at most one of its endpoints is matched.
            
            # This is a "matching with conflicts" or we need to constrain the matching.
            # Standard maximum matching on the endpoint graph does NOT guarantee that both endpoints of a path are not matched.
            # Example: Path A, Path B, Path C.
            # A can merge with B at A.start, B.start.
            # A can merge with C at A.end, C.end.
            # The auxiliary graph has nodes (A,0), (A,1), (B,0), (B,1), (C,0), (C,1).
            # Edges: ((A,0), (B,0)) and ((A,1), (C,1)).
            # A maximum matching might include both edges.
            # This would try to merge A with B and A with C.
            # This is physically impossible.
            
            # How to fix?
            # We can filter the matching to remove conflicting pairs.
            # Or, we can model it differently.
            # Since the number of paths is small, we can just iterate through the matching edges and keep the first valid one for each path.
            # Greedy selection from the maximum matching.
            
            used_paths = set()
            merges_to_apply = []
            
            # Sort matching edges? Maybe by some heuristic? 
            # Or just take all that don't conflict.
            # Since the matching is maximum, most edges are valid.
            # Conflicts only happen if a path is "double-booked".
            
            for (i, e1), (j, e2) in matching:
                if i in used_paths or j in used_paths:
                    continue
                # Check if this merge is still valid (it should be, as paths haven't changed yet)
                # We already know can_merge returned True when building the graph.
                merges_to_apply.append((i, j, e1, e2))
                used_paths.add(i)
                used_paths.add(j)
                
            if not merges_to_apply:
                break
                
            # Perform the merges
            new_paths = []
            # We need to keep paths that are not involved in any merge
            for k in range(n_paths):
                if k not in used_paths:
                    new_paths.append(paths[k])
                    
            for i, j, e1, e2 in merges_to_apply:
                merged = merge_paths(paths[i], paths[j], e1, e2)
                new_paths.append(merged)
                
            paths = new_paths
            changed = True

        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
