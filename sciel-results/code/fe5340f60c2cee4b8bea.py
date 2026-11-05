import time
import random
from collections import defaultdict, deque


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        edge_set = set(edges)
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == edge_set

    def eulerian_circuit(adj, start=None):
        """Find an Eulerian circuit in the graph (all vertices have even degree)."""
        if not adj:
            return []
        
        # Use Hierholzer's algorithm
        adj_dict = defaultdict(list)
        for u, v in adj.items():
            for w in v:
                adj_dict[u].append(w)
        
        # Make sure we have all vertices
        vertices = set()
        for u, v in adj.items():
            vertices.add(u)
            for w in v:
                vertices.add(w)
        
        for v in vertices:
            if v not in adj_dict:
                adj_dict[v] = []
        
        if start is None:
            # Find a vertex with odd degree if any, otherwise any vertex with edges
            for v in adj_dict:
                if adj_dict[v]:
                    start = v
                    break
            if start is None:
                return []
        
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            if adj_dict[v]:
                w = adj_dict[v].pop()
                stack.append(w)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def decompose_with_eulerian(edges, n, seed_val, time_limit):
        """
        Decompose edges into paths using Eulerian circuit approach.
        Add dummy edges to make all degrees even, find Eulerian circuit,
        then split at dummy edge locations.
        """
        if not edges:
            return []
        
        rng = random.Random(seed_val)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Count odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Already Eulerian
            circuit = eulerian_circuit(adj)
            if not circuit:
                return []
            # Remove consecutive duplicates
            path = []
            for v in circuit:
                if not path or path[-1] != v:
                    path.append(v)
            if len(path) > 1:
                return [path]
            return []
        
        # Pair up odd vertices
        odd_vertices = list(odd_vertices)
        rng.shuffle(odd_vertices)
        pairs = []
        for i in range(0, len(odd_vertices), 2):
            if i + 1 < len(odd_vertices):
                pairs.append((odd_vertices[i], odd_vertices[i + 1]))
        
        # Add dummy edges
        dummy_edges = set()
        for u, v in pairs:
            e = tuple(sorted((u, v)))
            if e not in set(tuple(sorted(e2)) for e2 in edges):
                dummy_edges.add(e)
            else:
                # If edge exists, we still mark it as a split point
                dummy_edges.add(e)
        
        # Build augmented graph
        adj_aug = defaultdict(list)
        for u, v in edges:
            adj_aug[u].append(v)
            adj_aug[v].append(u)
        for u, v in dummy_edges:
            adj_aug[u].append(v)
            adj_aug[v].append(u)
        
        # Find Eulerian circuit in augmented graph
        circuit = eulerian_circuit(adj_aug)
        if not circuit:
            return None
        
        # Remove consecutive duplicates
        path = []
        for v in circuit:
            if not path or path[-1] != v:
                path.append(v)
        path = path[:-1] if path and path[0] == path[-1] else path
        
        # Split at dummy edge locations
        # A dummy edge is between consecutive vertices in the circuit
        paths = []
        current_path = []
        for i in range(len(path)):
            current_path.append(path[i])
            if i + 1 < len(path):
                u, v = path[i], path[i + 1]
                e = tuple(sorted((u, v)))
                if e in dummy_edges:
                    if len(current_path) >= 2:
                        paths.append(current_path[:])
                    current_path = []
        
        if len(current_path) >= 2:
            paths.append(current_path)
        
        # Remove empty paths
        paths = [p for p in paths if len(p) >= 2]
        
        return paths if paths else None

    def greedy_extend(edges, k, seed_val, time_limit):
        """
        Greedy approach: start with k paths and extend them.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None
        
        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = [[] for _ in range(k)]
        
        edge_list = list(remaining)
        rng.shuffle(edge_list)
        
        start_edges = edge_list[:k]
        
        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]
        
        changed = True
        while changed and remaining:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)
            
            for i in idxs:
                if not paths[i]:
                    continue
                
                # Try extending back
                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)
                
                if candidates:
                    nb = rng.choice(candidates)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue
                
                # Try extending front
                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)
                
                if candidates:
                    nb = rng.choice(candidates)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue
        
        if remaining:
            return None
        return paths

    def merge_paths(paths, edges, seed_val, time_limit):
        """
        Try to merge paths that share endpoints to reduce path count.
        Two paths can be merged if one ends at a vertex where the other begins.
        """
        if not paths or len(paths) <= 1:
            return paths
        
        rng = random.Random(seed_val)
        # Build index: vertex -> list of (path_idx, end_type) where end_type is 0 for front, 1 for back
        end_index = defaultdict(list)
        for i, path in enumerate(paths):
            if not path:
                continue
            end_index[path[0]].append((i, 0))  # front
            end_index[path[-1]].append((i, 1))  # back
        
        merged = True
        while merged:
            merged = False
            # Try all pairs that can be merged
            for v in list(end_index.keys()):
                entries = end_index[v]
                if len(entries) < 2:
                    continue
                
                # Find a pair where one path ends at v and another starts at v
                # or one path starts at v and another ends at v
                front_entries = [e for e in entries if e[1] == 0]
                back_entries = [e for e in entries if e[1] == 1]
                
                for fi in front_entries:
                    for bi in back_entries:
                        if fi[0] == bi[0]:
                            continue
                        
                        # Check if merging is valid (no repeated vertices)
                        path_a = paths[fi[0]]
                        path_b = paths[bi[0]]
                        
                        # path_a starts at v, path_b ends at v
                        # Merged: path_b + path_a[1:]
                        merged_path = path_b + path_a[1:]
                        
                        if len(set(merged_path)) != len(merged_path):
                            continue
                        
                        # Check all edges are in original edge set
                        edge_set = set(tuple(sorted(e)) for e in edges)
                        valid = True
                        for j in range(len(merged_path) - 1):
                            e = tuple(sorted((merged_path[j], merged_path[j + 1])))
                            if e not in edge_set:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Merge
                        new_idx = fi[0]
                        old_idx = bi[0]
                        paths[new_idx] = merged_path
                        paths[old_idx] = []
                        
                        # Update index
                        end_index[merged_path[0]] = [(new_idx, 0)]
                        end_index[merged_path[-1]] = [(new_idx, 1)]
                        
                        # Clean up old entries
                        for key in list(end_index.keys()):
                            end_index[key] = [(pi, pt) for pi, pt in end_index[key] if pi not in (new_idx, old_idx) or (pi == new_idx and pt in (0, 1))]
                        
                        # Rebuild index properly
                        end_index = defaultdict(list)
                        for i, path in enumerate(paths):
                            if not path:
                                continue
                            end_index[path[0]].append((i, 0))
                            end_index[path[-1]].append((i, 1))
                        
                        merged = True
                        break
                    
                    if merged:
                        break
                
                if merged:
                    break
        
        return [p for p in paths if p]

    def solve_instance(n, p, seed, time_limit):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, (odd_count + 1) // 2)
        
        best_paths = None
        best_count = float('inf')
        
        inst_start = time.time()
        inst_deadline = inst_start + time_limit
        
        # Strategy 1: Eulerian approach with different pairings
        for restart in range(50):
            if time.time() > inst_deadline - 1:
                break
            seed_val = seed * 1000000 + restart
            paths = decompose_with_eulerian(edges, n, seed_val, inst_deadline - time.time())
            if paths is not None:
                if validate_paths(paths, edges):
                    # Try to merge
                    paths_merged = merge_paths(paths, edges, seed_val + 1, 0.1)
                    if validate_paths(paths_merged, edges):
                        if len(paths_merged) < best_count:
                            best_count = len(paths_merged)
                            best_paths = paths_merged
                    else:
                        if len(paths) < best_count:
                            best_count = len(paths)
                            best_paths = paths
        
        # Strategy 2: Greedy targeting
        for target_k in range(lower_bound, min(len(edges), lower_bound + 3) + 1):
            if time.time() > inst_deadline - 1:
                break
            for restart in range(2000):
                if time.time() > inst_deadline - 1:
                    break
                seed_val = seed * 100000 + restart
                paths = greedy_extend(edges, target_k, seed_val, inst_deadline - time.time())
                if paths is not None:
                    if validate_paths(paths, edges):
                        if target_k < best_count:
                            best_count = target_k
                            best_paths = paths
                        break
        
        if best_paths is None:
            # Fallback: greedy without target
            rng = random.Random(seed + 9999)
            remaining = set(edges)
            adj2 = defaultdict(set)
            for u, v in edges:
                adj2[u].add(v)
                adj2[v].add(u)
            
            paths = []
            while remaining:
                degs = {}
                for u, v in remaining:
                    degs[u] = degs.get(u, 0) + 1
                    degs[v] = degs.get(v, 0) + 1
                if not degs:
                    break
                max_d = max(degs.values())
                cands = [v for v, d in degs.items() if d == max_d]
                start_u = rng.choice(cands)
                
                neighbors = [nb for nb in adj2[start_u]
                             if (start_u, nb) in remaining or (nb, start_u) in remaining]
                if not neighbors:
                    e = next(iter(remaining))
                    start_u, start_v = e
                else:
                    start_v = rng.choice(neighbors)
                
                path = [start_u, start_v]
                if (start_u, start_v) in remaining:
                    remaining.remove((start_u, start_v))
                else:
                    remaining.remove((start_v, start_u))
                
                changed = True
                while changed:
                    changed = False
                    back = path[-1]
                    cands = [nb for nb in adj2[back]
                             if nb not in path and
                             ((back, nb) in remaining or (nb, back) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (back, nb) in remaining:
                            remaining.remove((back, nb))
                        else:
                            remaining.remove((nb, back))
                        path.append(nb)
                        changed = True
                        continue
                    front = path[0]
                    cands = [nb for nb in adj2[front]
                             if nb not in path and
                             ((front, nb) in remaining or (nb, front) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (front, nb) in remaining:
                            remaining.remove((front, nb))
                        else:
                            remaining.remove((nb, front))
                        path.insert(0, nb)
                        changed = True
                        continue
                
                paths.append(path)
            
            return paths
        
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 15:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        time_per_instance = min(40.0, deadline - time.time() - 10)
        paths = solve_instance(n, p, seed, time_per_instance)
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
