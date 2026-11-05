import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        n_vertices = max(max(u, v) for u, v in edges) + 1
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        edge_set = set(edges)
        
        # Build list of edges and map
        edge_list = list(edge_set)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        
        # Incident edges for each vertex
        vertex_edges = defaultdict(list)
        for u, v in edge_list:
            vertex_edges[u].append((u, v))
            vertex_edges[v].append((v, u))
            
        # State: pairing of edges at each vertex
        # pairing[v] is a list of pairs of edge indices (or None for unpaired)
        # Actually, easier to represent as a permutation of incident edges where (2i, 2i+1) are paired.
        # If degree is odd, one is unpaired.
        
        def get_paths(pairing):
            """
            pairing: dict vertex -> list of incident edge indices, ordered such that 
            indices 0,1 are paired, 2,3 are paired, etc.
            Returns list of paths (list of vertices) or None if invalid (non-simple).
            """
            # Build next-edge map
            # For each vertex v and incident edge e1, find the paired edge e2.
            # The path traverses e1 into v, then e2 out of v.
            
            visited_edges = set()
            paths = []
            
            # Map from (vertex, edge_index) to next (vertex, edge_index)
            # Actually, we just need to follow the chain.
            
            # For each edge, determine its two endpoints and which edge it is paired with at each end.
            # Let's represent the state as:
            # next_edge[v][e_idx] = e_idx2, where e_idx is an edge incident to v, and e_idx2 is the edge paired with it at v.
            # If e_idx is unpaired, next_edge[v][e_idx] = -1.
            
            next_edge = {}
            for v in range(n_vertices):
                inc_edges = vertex_edges[v]
                p_list = pairing[v]
                # p_list is a permutation of indices in inc_edges
                # Pair (p_list[0], p_list[1]), (p_list[2], p_list[3])...
                for i in range(0, len(p_list), 2):
                    if i + 1 < len(p_list):
                        e1_idx = p_list[i]
                        e2_idx = p_list[i+1]
                        # e1_idx and e2_idx are indices into vertex_edges[v]
                        e1 = inc_edges[e1_idx]
                        e2 = inc_edges[e2_idx]
                        # The edge in the global list is tuple(sorted(e1))
                        ge1 = tuple(sorted(e1))
                        ge2 = tuple(sorted(e2))
                        # At vertex v, edge ge1 is connected to ge2
                        next_edge[(v, ge1)] = ge2
                        next_edge[(v, ge2)] = ge1
                    else:
                        e1_idx = p_list[i]
                        e1 = inc_edges[e1_idx]
                        ge1 = tuple(sorted(e1))
                        next_edge[(v, ge1)] = None # Unpaired

            # Now trace paths
            # Start from any unpaired edge end
            for v in range(n_vertices):
                for ge in vertex_edges[v]:
                    ge_sorted = tuple(sorted(ge))
                    if (v, ge_sorted) in next_edge and next_edge[(v, ge_sorted)] is None:
                        # This is a path start
                        if ge_sorted in visited_edges:
                            continue
                            
                        path_vertices = []
                        cur_v = v
                        cur_e = ge_sorted
                        is_valid = True
                        
                        while cur_e is not None:
                            if cur_e in visited_edges:
                                # Cycle detected or edge reuse
                                is_valid = False
                                break
                            visited_edges.add(cur_e)
                            
                            u, w = cur_e
                            # Determine which vertex is the "other" end
                            if cur_v == u:
                                next_v = w
                            else:
                                next_v = u
                                
                            if not path_vertices:
                                path_vertices.append(cur_v)
                            path_vertices.append(next_v)
                            
                            if next_v in path_vertices[:-1]: # Check if next_v is already in path (simple path check)
                                # If next_v is the start, it's a cycle (invalid for path unless we allow, but we want paths)
                                # Actually, if next_v is in path_vertices, the path is not simple.
                                is_valid = False
                                break
                                
                            # Move to next edge
                            cur_v = next_v
                            cur_e = next_edge.get((cur_v, cur_e), None)
                            
                        if is_valid:
                            paths.append(path_vertices)
            
            # Check if all edges are covered
            if len(visited_edges) != len(edge_list):
                return None # Some edges not covered or cycles
            
            return paths

        # Initialize pairing randomly
        def random_pairing():
            pairing = {}
            for v in range(n_vertices):
                inc_edges = vertex_edges[v]
                indices = list(range(len(inc_edges)))
                random.shuffle(indices)
                pairing[v] = indices
            return pairing

        best_paths = None
        best_score = float('inf')
        
        # Initial greedy solution
        # Try a few random initial pairings
        for _ in range(10):
            p = random_pairing()
            paths = get_paths(p)
            if paths:
                score = len(paths)
                if score < best_score:
                    best_score = score
                    best_paths = paths
                    best_pairing = p.copy()
        
        if best_paths is None:
             # Fallback: find a valid path decomposition using DFS
             # This is expensive but necessary if SA fails to find valid
             # For now, return a trivial decomposition
             return [[u, v] for u, v in edges]

        # Simulated Annealing
        current_pairing = best_pairing
        current_paths = get_paths(current_pairing)
        current_score = len(current_paths)
        
        T = 1.0
        cooling = 0.995
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random vertex and swap two elements in its pairing
            v = random.randint(0, n_vertices - 1)
            p_list = current_pairing[v]
            if len(p_list) < 2:
                continue
            i, j = random.sample(range(len(p_list)), 2)
            
            # Swap
            p_list[i], p_list[j] = p_list[j], p_list[i]
            
            new_paths = get_paths(current_pairing)
            if new_paths:
                new_score = len(new_paths)
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_score = new_score
                    current_paths = new_paths
                    if current_score < best_score:
                        best_score = current_score
                        best_paths = current_paths
                        best_pairing = [lst[:] for lst in current_pairing.values()]
                else:
                    # Revert
                    p_list[i], p_list[j] = p_list[j], p_list[i]
            else:
                # Invalid state, revert
                p_list[i], p_list[j] = p_list[j], p_list[i]
                
            T *= cooling
            
        return best_paths

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        for path in paths:
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        if used_edges != edge_set:
            return False
        return True

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_graph(edges)
        
        if not validate_paths(paths, edges):
            paths = [[u, v] for u, v in edges]
            
        results.append(paths)
        
        if time.time() > deadline - 1:
            break
            
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
