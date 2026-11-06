import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s limit
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate random bounded degree graph
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_bitmasks(edges, n):
        """Build conflict bitmasks for strong edge coloring.
        Two edges conflict if they share a vertex OR are connected by an edge."""
        num_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Adjacency for original graph (for edge-existence checks)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, compute bitmask of conflicting edges
        conflict_masks = [0] * num_edges
        
        for i, (u, v) in enumerate(edges):
            mask = 0
            for j, (x, y) in enumerate(edges):
                if i == j:
                    continue
                # Check if edges conflict:
                # 1. Share a vertex
                if u == x or u == y or v == x or v == y:
                    mask |= (1 << j)
                    continue
                # 2. Connected by an edge in original graph
                if (min(u, x), max(u, x)) in edge_to_idx:
                    mask |= (1 << j)
                    continue
                if (min(u, y), max(u, y)) in edge_to_idx:
                    mask |= (1 << j)
                    continue
                if (min(v, x), max(v, x)) in edge_to_idx:
                    mask |= (1 << j)
                    continue
                if (min(v, y), max(v, y)) in edge_to_idx:
                    mask |= (1 << j)
                    continue
            conflict_masks[i] = mask
        
        return conflict_masks
    
    def max_induced_matching_from_vertex(conflict_masks, start_vertex, used_mask):
        """Find a maximal induced matching (clique in complement) starting from start_vertex.
        Returns (set_of_vertices, mask_of_vertices)."""
        # A clique in the complement graph = set of pairwise non-conflicting edges
        # In terms of conflict_masks: v can join clique if (conflict_masks[v] & clique_mask) == 0
        clique = {start_vertex}
        clique_mask = (1 << start_vertex)
        
        # Greedy: repeatedly add vertex with fewest conflicts with current clique
        while True:
            best_v = -1
            best_score = -1
            for v in range(len(conflict_masks)):
                if v in clique or (clique_mask >> v) & 1:
                    continue
                # Check if v is compatible with all vertices in clique
                if conflict_masks[v] & clique_mask:
                    continue
                # Compatible! Score by degree (prefer high-degree for better coverage)
                score = len(conflict_masks[v].bin_count() if hasattr(conflict_masks[v], 'bin_count') else bin(conflict_masks[v]).count('1'))
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            clique.add(best_v)
            clique_mask |= (1 << best_v)
        
        return clique, clique_mask
    
    def greedy_set_cover_with_refinement(conflict_masks, time_budget):
        """Find minimum clique cover of conflict graph using greedy set-cover with local refinement.
        
        Each color class is a clique in the complement of the conflict graph (induced matching).
        We want to cover all vertices with minimum number of such cliques.
        """
        num_vertices = len(conflict_masks)
        
        # Phase 1: Greedy set cover
        # Repeatedly pick the clique that covers the most uncovered vertices
        uncovered = set(range(num_vertices))
        cliques = []  # List of (clique_set, clique_mask)
        
        while uncovered:
            # Find best clique: maximize coverage of uncovered vertices
            best_clique = None
            best_coverage = -1
            best_clique_mask = 0
            
            # Try starting from each uncovered vertex
            for start_v in sorted(uncovered, key=lambda v: len(conflict_masks[v]), reverse=True):
                clique, clique_mask = max_induced_matching_from_vertex(conflict_masks, start_v, 0)
                coverage = len(clique & uncovered)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = clique
                    best_clique_mask = clique_mask
            
            if best_clique is None or best_coverage == 0:
                # Shouldn't happen, but safety
                break
            
            cliques.append((best_clique, best_clique_mask))
            uncovered -= best_clique
        
        # Phase 2: Local refinement - try to merge or reassign vertices to reduce clique count
        # Try removing one vertex from a clique and reassigning it to another
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            # Try moving vertices between cliques
            for i in range(len(cliques)):
                clique_i, mask_i = cliques[i]
                for v in list(clique_i):
                    # Try removing v from clique_i and adding to another clique
                    new_clique_i = clique_i - {v}
                    for j in range(len(cliques)):
                        if j == i:
                            continue
                        clique_j, mask_j = cliques[j]
                        # Check if v is compatible with all in clique_j
                        compatible = True
                        for u in clique_j:
                            if conflict_masks[v] & (1 << u):
                                compatible = False
                                break
                        if compatible:
                            # Move v from i to j
                            cliques[i] = (new_clique_i, cliques[i][1] & ~(1 << v))
                            cliques[j] = (clique_j | {v}, cliques[j][1] | (1 << v))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Try merging two cliques if possible
            for i in range(len(cliques)):
                for j in range(i+1, len(cliques)):
                    clique_i, mask_i = cliques[i]
                    clique_j, mask_j = cliques[j]
                    # Check if union is still a valid clique (induced matching)
                    valid = True
                    for a in clique_i:
                        for b in clique_j:
                            if conflict_masks[a] & (1 << b):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Merge
                        merged = clique_i | clique_j
                        merged_mask = mask_i | mask_j
                        cliques[i] = (merged, merged_mask)
                        del cliques[j]
                        improved = True
                        break
                if improved:
                    break
        
        # Assign colors: each clique gets a color
        coloring = [-1] * num_vertices
        for color_idx, (clique, mask) in enumerate(cliques):
            for v in clique:
                coloring[v] = color_idx
        
        return coloring
    
    def greedy_coloring(conflict_masks):
        """Simple greedy coloring as fallback."""
        num_vertices = len(conflict_masks)
        coloring = [-1] * num_vertices
        # Order by degree (descending)
        order = sorted(range(num_vertices), key=lambda i: bin(conflict_masks[i]).count('1'), reverse=True)
        
        for v in order:
            # Find available colors
            used_colors = set()
            mask = conflict_masks[v]
            for u in range(num_vertices):
                if (mask >> u) & 1 and coloring[u] != -1:
                    used_colors.add(coloring[u])
            
            c = 0
            while c in used_colors:
                c += 1
            coloring[v] = c
        
        return coloring
    
    def verify_coloring(coloring, conflict_masks):
        """Verify that each color class is an induced matching."""
        num_vertices = len(conflict_masks)
        color_classes = defaultdict(list)
        for v in range(num_vertices):
            color_classes[coloring[v]].append(v)
        
        for color, vertices in color_classes.items():
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    a, b = vertices[i], vertices[j]
                    # They must NOT conflict
                    if conflict_masks[a] & (1 << b):
                        return False
        return True
    
    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time per instance
    time_per_instance = (time_limit - 10) / len(specs)
    
    for idx, (n, d, seed) in enumerate(specs):
        edges = instance(n, d, seed)
        conflict_masks = build_conflict_bitmasks(edges, n)
        
        # Use greedy set cover with refinement
        coloring = greedy_set_cover_with_refinement(conflict_masks, time_limit - 5)
        
        # Verify
        if not verify_coloring(coloring, conflict_masks):
            # Fallback to greedy
            coloring = greedy_coloring(conflict_masks)
        
        results.append(coloring)
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            # If running out of time, fill remaining with greedy
            for remaining_idx in range(idx+1, len(specs)):
                rn, rd, rseed = specs[remaining_idx]
                redges = instance(rn, rd, rseed)
                rconflict = build_conflict_bitmasks(redges, rn)
                rcoloring = greedy_coloring(rconflict)
                results.append(rcoloring)
            break
    
    return results
