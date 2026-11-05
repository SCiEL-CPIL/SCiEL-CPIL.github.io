import time
import random
from collections import defaultdict, deque

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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
                continue
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    def eulerian_cycle(adj, start, rng):
        """
        Find an Eulerian cycle in a directed graph using Hierholzer's algorithm.
        adj: dict of list of neighbors (directed)
        start: starting vertex
        rng: random generator for shuffling adjacency lists
        Returns: list of vertices in the cycle
        """
        # Make a copy of adj and shuffle to allow different cycles
        adj_copy = {u: list(v) for u, v in adj.items()}
        for u in adj_copy:
            rng.shuffle(adj_copy[u])
        
        stack = [start]
        path = []
        
        while stack:
            u = stack[-1]
            if adj_copy[u]:
                v = adj_copy[u].pop()
                stack.append(v)
            else:
                path.append(stack.pop())
                
        return path[::-1]

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degrees
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        
        # Theoretical Lower Bound
        if num_odd == 0:
            # If no odd vertices, we can form an Eulerian circuit.
            # But we need simple paths.
            # We can break the circuit into paths.
            # However, if the graph is a single cycle, we need 2 paths (one for the cycle minus one edge, one for the edge).
            # Actually, we can just use the Eulerian cycle approach with 0 virtual edges.
            # The result will be 1 cycle. We need to convert it to paths.
            # A cycle v0-v1-...-vk-v0 can be split into path v0-...-vk and edge vk-v0? No.
            # If we output the cycle as a path [v0, ..., vk, v0], it's not simple.
            # So we must split it.
            # Let's try to find a decomposition into simple paths.
            # For now, let's handle num_odd > 0 with the pairing method.
            # If num_odd == 0, we can add 2 virtual edges between two arbitrary vertices to create 2 odd vertices?
            # No, that changes the graph.
            # Let's just try to find simple paths covering all edges.
            # Greedy might work well here, or we can just return a valid decomposition.
            pass

        best_paths = None
        best_count = float('inf')
        
        # Strategy: Pair odd vertices, add virtual edges, find Eulerian cycle, remove virtual edges.
        # The number of paths will be num_odd // 2.
        
        max_attempts = 500
        rng_master = random.Random(seed)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            # 1. Pair odd vertices
            if num_odd > 0:
                # Shuffle odd vertices to get different pairings
                odd_copy = odd_vertices[:]
                rng_master.shuffle(odd_copy)
                pairs = list(zip(odd_copy[0::2], odd_copy[1::2]))
            else:
                # If no odd vertices, we can't use this method directly.
                # We can pick two vertices and treat them as "odd" by adding virtual edges?
                # No, that would add paths.
                # If num_odd == 0, the lower bound is 1 (if connected).
                # But we need simple paths.
                # Let's fallback to greedy for this case or handle separately.
                # For now, skip this attempt if num_odd == 0 to avoid complexity, 
                # or use a dummy pairing if we allow it.
                # Actually, if num_odd == 0, we can just use the greedy approach from the seed program.
                continue

            # 2. Create augmented graph
            # We need to simulate the virtual edges.
            # We will build a directed adjacency list for Hierholzer.
            # For each real edge (u,v), we add u->v and v->u.
            # For each virtual edge (u,v), we add u->v and v->u.
            # We mark virtual edges to remove them later.
            
            adj_dir = defaultdict(list)
            # Track which edges are virtual
            # We need to be able to remove them from the cycle.
            # Let's store the cycle as a list of (vertex, edge_id) or just vertices and check if the edge is virtual.
            
            # To make it easier, let's just build the list of edges for the augmented graph.
            # Each edge is (u, v, is_virtual)
            
            aug_edges = []
            for u, v in edges:
                aug_edges.append((u, v, False))
                aug_edges.append((v, u, False))
            
            for u, v in pairs:
                aug_edges.append((u, v, True))
                aug_edges.append((v, u, True))
            
            # Build adjacency for Hierholzer
            # We need to consume edges.
            # adj_dir[u] = list of (v, edge_index)
            
            for idx, (u, v, is_v) in enumerate(aug_edges):
                adj_dir[u].append((v, idx))
                
            # Shuffle adjacency lists for randomization
            for u in adj_dir:
                rng_master.shuffle(adj_dir[u])
            
            # 3. Find Eulerian Cycle
            # Start at any vertex. Let's pick odd_vertices[0] if exists.
            start_v = odd_vertices[0] if num_odd > 0 else 0
            
            # Hierholzer
            stack = [start_v]
            path_vertices = []
            path_edge_ids = []
            
            while stack:
                u = stack[-1]
                if adj_dir[u]:
                    v, e_idx = adj_dir[u].pop()
                    stack.append(v)
                    # We need to remember which edge we used to get to v
                    # But we pop it, so we can't easily trace back which edge led to which vertex in the final path
                    # Standard Hierholzer returns vertices. We need to know the edges.
                    # We can reconstruct edges by looking at consecutive vertices in the path.
                    # But we need to know if that edge was virtual.
                    # Since the graph is simple (no multi-edges in original), and virtual edges are unique per pair,
                    # we can just check if the edge (u,v) is in the virtual set.
                    
                else:
                    path_vertices.append(stack.pop())
            
            path_vertices = path_vertices[::-1]
            
            if len(path_vertices) != len(aug_edges) + 1:
                # Cycle not found or incorrect
                continue
            
            # 4. Convert cycle to paths
            # The cycle is a sequence of vertices.
            # We remove the virtual edges.
            # A virtual edge is an edge (u,v) where {u,v} is in the set of pairs.
            
            virtual_edge_set = set()
            for u, v in pairs:
                virtual_edge_set.add(tuple(sorted((u, v))))
            
            paths = []
            current_path = []
            
            for i in range(len(path_vertices) - 1):
                u = path_vertices[i]
                v = path_vertices[i+1]
                e = tuple(sorted((u, v)))
                
                if not current_path:
                    current_path.append(u)
                
                current_path.append(v)
                
                if e in virtual_edge_set:
                    # End of a path
                    if len(current_path) >= 2:
                        # Check if simple
                        if len(set(current_path)) == len(current_path):
                            paths.append(current_path)
                        else:
                            # Path not simple, discard this attempt
                            paths = None
                            break
                    current_path = []
            
            # The last segment after the last virtual edge
            if paths is not None:
                if current_path:
                    if len(current_path) >= 2:
                        if len(set(current_path)) == len(current_path):
                            paths.append(current_path)
                        else:
                            paths = None
                # Also, the cycle closes, so the last vertex is the same as the first.
                # The path from the last virtual edge to the start is the last path.
                # Wait, the cycle is v0...vk=v0.
                # We iterate i from 0 to k-1.
                # The edge (vk, v0) is handled?
                # The loop goes to len(path_vertices)-1.
                # path_vertices has k+1 elements.
                # Edges are (v0,v1), ..., (vk-1, vk).
                # The edge (vk, v0) is missing from the loop if we only go to len-1.
                # But path_vertices[-1] == path_vertices[0].
                # So the last edge is (vk, v0).
                # We need to handle the closure.
                
                # Actually, the loop `for i in range(len(path_vertices) - 1)` covers edges 0 to k-1.
                # The edge k (vk to v0) is not covered.
                # But since it's a cycle, the "last" path is from the last virtual edge to the start.
                # Let's re-verify the logic.
                
                # If we break on virtual edges, we get segments.
                # The last segment ends at v0 (which is the start).
                # So `current_path` will contain the vertices from the last virtual edge to v0.
                # We add it to paths.
                
                # Check if all paths are valid
                if paths:
                    # Validate that these paths cover all original edges
                    # and are edge-disjoint.
                    # Since they come from an Eulerian cycle of the augmented graph,
                    # they cover all augmented edges exactly once.
                    # Removing virtual edges leaves original edges covered exactly once.
                    # So we just need to check simplicity (already done) and validity.
                    
                    if validate_paths(paths, edges):
                        if len(paths) < best_count:
                            best_count = len(paths)
                            best_paths = paths
                            
                            # If we reached the lower bound, we can stop
                            if len(paths) <= num_odd // 2:
                                break

        # Fallback to greedy if pairing didn't work
        if best_paths is None:
            # Use a simple greedy approach
            rng = random.Random(seed + 999)
            remaining = set(edges)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            paths = []
            while remaining:
                edge = rng.choice(list(remaining))
                remaining.remove(edge)
                u, v = edge
                path = [u, v]
                
                changed = True
                while changed:
                    changed = False
                    for end in [0, 1]:
                        if end == 0:
                            current = path[0]
                            candidates = [nb for nb in adj[current] 
                                          if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                            if candidates:
                                nb = rng.choice(candidates)
                                if (current, nb) in remaining:
                                    remaining.remove((current, nb))
                                else:
                                    remaining.remove((nb, current))
                                path.insert(0, nb)
                                changed = True
                        else:
                            current = path[-1]
                            candidates = [nb for nb in adj[current] 
                                          if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                            if candidates:
                                nb = rng.choice(candidates)
                                if (current, nb) in remaining:
                                    remaining.remove((current, nb))
                                else:
                                    remaining.remove((nb, current))
                                path.append(nb)
                                changed = True
                paths.append(path)
            
            if validate_paths(paths, edges):
                best_paths = paths

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    # Allocate time
    time_alloc = [35, 40, 40]
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 5:
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
