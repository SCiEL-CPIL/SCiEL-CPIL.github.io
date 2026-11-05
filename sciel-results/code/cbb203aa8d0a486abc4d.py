import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0

        edge_set = set(edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                # A single vertex path covers no edges, but is it allowed?
                # Task says "lists of paths (vertex lists)". Usually paths have at least 1 edge.
                # "cover all edges". If a path has 1 vertex, it covers 0 edges.
                # If there are edges, we must cover them.
                # If no edges, empty list or list of single vertices?
                # The fallback in the original code returned [[u,v] for u,v in edges].
                # I will assume paths must have at least 2 vertices to cover an edge.
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)

        return used_edges == edge_set

    def greedy_decompose(edges, adj, vertices, rng):
        """
        Decompose edges into simple paths using a greedy randomized walk.
        Returns a list of paths.
        """
        remaining = set(edges)
        paths = []
        
        # Pre-compute adjacency for quick lookup
        # adj is a dict of sets of neighbors
        
        # Sort edges to ensure determinism if needed, but we use rng
        edge_list = list(remaining)
        
        while remaining:
            # Pick a random starting edge to ensure we don't get stuck in a loop of bad choices
            # Or pick a random vertex with remaining degree > 0
            active_vertices = [v for v in vertices if any(e in remaining for e in adj[v])]
            if not active_vertices:
                break
                
            start_v = rng.choice(active_vertices)
            
            # Pick a random neighbor to start the path
            neighbors = [nb for nb in adj[start_v] if (start_v, nb) in remaining or (nb, start_v) in remaining]
            if not neighbors:
                continue
                
            next_v = rng.choice(neighbors)
            
            # Start path
            path = [start_v, next_v]
            e = tuple(sorted((start_v, next_v)))
            remaining.remove(e)
            
            path_vertices = set(path)
            
            # Extend the path greedily
            extended = True
            while extended:
                extended = False
                # Try to extend at the end
                end_v = path[-1]
                end_neighbors = [nb for nb in adj[end_v] 
                                  if nb not in path_vertices 
                                  and ((end_v, nb) in remaining or (nb, end_v) in remaining)]
                if end_neighbors:
                    nb = rng.choice(end_neighbors)
                    e = tuple(sorted((end_v, nb)))
                    remaining.remove(e)
                    path.append(nb)
                    path_vertices.add(nb)
                    extended = True
                    continue
                
                # Try to extend at the beginning
                start_v = path[0]
                start_neighbors = [nb for nb in adj[start_v] 
                                   if nb not in path_vertices 
                                   and ((start_v, nb) in remaining or (nb, start_v) in remaining)]
                if start_neighbors:
                    nb = rng.choice(start_neighbors)
                    e = tuple(sorted((start_v, nb)))
                    remaining.remove(e)
                    path.insert(0, nb)
                    path_vertices.add(nb)
                    extended = True
                    continue
            
            paths.append(path)
            
        return paths

    def try_merge(paths, adj, edges_set, rng):
        """
        Try to merge two paths if they share an endpoint and the result is simple.
        Returns True if a merge was performed, modifying paths in place.
        """
        n_paths = len(paths)
        if n_paths < 2:
            return False
        
        # Create a list of indices to shuffle
        indices = list(range(n_paths))
        rng.shuffle(indices)
        
        for i_idx in indices:
            p1 = paths[i_idx]
            if len(p1) < 2:
                continue
            v1_start = p1[0]
            v1_end = p1[-1]
            set1 = set(p1)
            
            for j_idx in indices:
                if i_idx == j_idx:
                    continue
                p2 = paths[j_idx]
                if len(p2) < 2:
                    continue
                v2_start = p2[0]
                v2_end = p2[-1]
                
                # Check if they can be merged
                # Case 1: p1 ends at v, p2 starts at v
                if v1_end == v2_start:
                    # New path: p1 + p2[1:]
                    # Check if p2[1:] has any vertex in p1
                    # Actually, just check if the union is simple
                    # p1 + p2[1:]
                    # Vertices in p1: set1
                    # Vertices in p2[1:]: set(p2) - {v2_start}
                    # Since v1_end == v2_start, the connection is valid.
                    # We need to ensure no vertex in p2[1:] is in p1.
                    if not (set(p2[1:]) & set1):
                        new_path = p1 + p2[1:]
                        # Replace p1 and p2 with new_path
                        paths[i_idx] = new_path
                        paths[j_idx] = None
                        return True
                
                # Case 2: p1 starts at v, p2 ends at v
                if v1_start == v2_end:
                    # New path: p2[::-1] + p1[1:] ? No.
                    # p2 is (a, ..., v). p1 is (v, ..., b).
                    # Merged: (a, ..., v, ..., b)
                    # p2 + p1[1:]
                    if not (set(p1[1:]) & set(p2)):
                        new_path = p2 + p1[1:]
                        paths[i_idx] = new_path
                        paths[j_idx] = None
                        return True

                # Case 3: p1 ends at v, p2 ends at v
                # We can reverse p2: (v, ..., a)
                # Then merge p1 + reversed(p2)[1:]
                if v1_end == v2_end:
                    rev_p2 = p2[::-1]
                    # rev_p2 starts at v
                    if not (set(rev_p2[1:]) & set1):
                        new_path = p1 + rev_p2[1:]
                        paths[i_idx] = new_path
                        paths[j_idx] = None
                        return True

                # Case 4: p1 starts at v, p2 starts at v
                # Reverse p1: (v, ..., a)
                # Merge p2 + reversed(p1)[1:]
                if v1_start == v2_start:
                    rev_p1 = p1[::-1]
                    if not (set(rev_p1[1:]) & set(p2)):
                        new_path = p2 + rev_p1[1:]
                        paths[i_idx] = new_path
                        paths[j_idx] = None
                        return True

        return False

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        vertices = sorted(adj.keys())
        
        # Compute lower bound
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, odd_count // 2) if odd_count > 0 else 1
        
        best_paths = None
        best_count = float('inf')
        
        # Time allocation
        inst_start = time.time()
        inst_deadline = inst_start + 35.0
        
        for restart in range(5000):
            if time.time() > inst_deadline:
                break
            
            rng = random.Random(seed * 100000 + restart)
            
            # 1. Greedy Decomposition
            paths = greedy_decompose(edges, adj, vertices, rng)
            
            # Remove None paths if any (shouldn't happen in greedy)
            paths = [p for p in paths if p]
            
            # 2. Iterative Merging
            merged = True
            while merged:
                merged = try_merge(paths, adj, set(edges), rng)
                # Filter out None paths after merge
                paths = [p for p in paths if p is not None]
                
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
                
                # Early exit if we hit the lower bound
                if best_count == lower_bound:
                    break

        if best_paths is None:
            # Fallback: every edge is a path
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)

    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
