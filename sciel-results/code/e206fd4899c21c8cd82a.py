"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    if m == 0:
        return []

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Edge list as frozenset for O(1) lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))

    # Precompute neighbor lists
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # We'll use a greedy approach with bitmask-based biclique finding
    # Key idea: find maximal bicliques in the remaining graph, greedily
    # covering edges

    remaining_edges = set(edge_set)
    remaining_adj = [adj[i] for i in range(n)]  # bitmask of remaining neighbors
    remaining_deg = [0] * n
    for u, v in remaining_edges:
        remaining_deg[u] += 1
        remaining_deg[v] += 1

    bicliques = []
    deadline = time.time() + 75.0  # Safety margin

    while remaining_edges and time.time() < deadline:
        # Find the best biclique to extract
        # Strategy: try to find a large biclique by expanding from an edge
        # Pick an edge with high "potential" (high degree endpoints)
        
        # Sort remaining edges by sum of degrees
        edge_list = list(remaining_edges)
        if not edge_list:
            break
        
        # Try several candidate starting edges
        best_biclique = None
        best_biclique_edges = 0

        # Candidate edges: pick from high-degree vertices
        # Get all edges, sort by combined degree
        scored_edges = []
        for (u, v) in edge_list:
            scored_edges.append((remaining_deg[u] + remaining_deg[v], u, v))
        scored_edges.sort(reverse=True)

        # Try top candidates
        candidates_to_try = scored_edges[:min(30, len(scored_edges))]
        
        for _, u0, v0 in candidates_to_try:
            # Try to build a biclique containing edge (u0, v0)
            # Start with left={u0}, right={v0}
            # Expand: add vertices that are connected to ALL vertices on the other side
            
            # We'll do a greedy expansion
            left = {u0}
            right = {v0}
            
            # Find common neighbors for expansion
            # A vertex w can be added to left if it's connected to all of right
            # A vertex w can be added to right if it's connected to all of left
            
            expanded = True
            while expanded:
                expanded = False
                
                # Try to add to left: vertices connected to all in right
                right_mask = 0
                for r in right:
                    right_mask |= remaining_adj[r]
                
                # Candidates for left: vertices in right_mask that are not in left
                # and are connected to all in right
                for w in range(n):
                    if w in left:
                        continue
                    if (right_mask >> w) & 1:
                        # w is connected to all in right
                        # Check w is connected to all in left (should be by construction)
                        can_add = True
                        for l in left:
                            if not ((remaining_adj[l] >> w) & 1):
                                can_add = False
                                break
                        if can_add:
                            left.add(w)
                            expanded = True
                            break  # Restart to be safe
                
                # Try to add to right: vertices connected to all in left
                left_mask = 0
                for l in left:
                    left_mask |= remaining_adj[l]
                
                for w in range(n):
                    if w in right:
                        continue
                    if (left_mask >> w) & 1:
                        can_add = True
                        for r in right:
                            if not ((remaining_adj[r] >> w) & 1):
                                can_add = False
                                break
                        if can_add:
                            right.add(w)
                            expanded = True
                            break
            
            # Count edges in this biclique
            biclique_edges = set()
            for l in left:
                for r in right:
                    e = (l, r) if l < r else (r, l)
                    if e in remaining_edges:
                        biclique_edges.add(e)
            
            if len(biclique_edges) > best_biclique_edges:
                best_biclique_edges = len(biclique_edges)
                best_biclique = (left, right, biclique_edges)

        if best_biclique is None or best_biclique_edges == 0:
            # Fallback: just take a single edge
            u0, v0 = edge_list[0]
            best_biclique = ({u0}, {v0}, {(u0, v0) if u0 < v0 else (v0, u0)})
            best_biclique_edges = 1

        left, right, biclique_edges = best_biclique
        
        # Remove these edges from remaining
        for e in biclique_edges:
            remaining_edges.discard(e)
            u, v = e
            remaining_adj[u] &= ~(1 << v)
            remaining_adj[v] &= ~(1 << u)
            remaining_deg[u] -= 1
            remaining_deg[v] -= 1

        bicliques.append(([int(x) for x in sorted(left)], [int(x) for x in sorted(right)]))

    # Fallback: if we ran out of time, add remaining edges as single-edge bicliques
    for (u, v) in remaining_edges:
        bicliques.append([[int(u)], [int(v)]])

    return bicliques
# EVOLVE-BLOCK-END
