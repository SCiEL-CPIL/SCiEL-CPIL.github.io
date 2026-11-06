import time
import random
from collections import defaultdict, deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def hopcroft_karp(left_nodes, right_nodes, edges):
        """
        left_nodes: list of left node ids
        right_nodes: list of right node ids
        edges: list of (u, v) where u in left_nodes, v in right_nodes
        Returns: set of matched pairs (u, v)
        """
        # Map node ids to indices for faster access
        left_index = {u: i for i, u in enumerate(left_nodes)}
        right_index = {v: i for i, v in enumerate(right_nodes)}
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Adjacency list for left nodes: list of right node indices
        adj = [[] for _ in range(n_left)]
        for u, v in edges:
            if u in left_index and v in right_index:
                adj[left_index[u]].append(right_index[v])
        
        # Matching arrays
        # pair_u[i] = index of right node matched to left node i, or -1
        pair_u = [-1] * n_left
        # pair_v[j] = index of left node matched to right node j, or -1
        pair_v = [-1] * n_right
        
        def bfs():
            """
            Returns True if there is an augmenting path.
            dist[i] = distance from left node i to the NIL node in the layered graph.
            """
            dist = [-1] * n_left
            queue = deque()
            
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            
            found = False
            while queue:
                u = queue.popleft()
                for v_idx in adj[u]:
                    # v_idx is index in right_nodes
                    u_prime = pair_v[v_idx]
                    if u_prime != -1 and dist[u_prime] == -1:
                        dist[u_prime] = dist[u] + 1
                        queue.append(u_prime)
                    elif u_prime == -1:
                        found = True
            return found

        def dfs(u):
            for v_idx in adj[u]:
                u_prime = pair_v[v_idx]
                if u_prime == -1 or (dist[u_prime] == dist[u] + 1 and dfs(u_prime)):
                    pair_u[u] = v_idx
                    pair_v[v_idx] = u
                    return True
            dist[u] = -1
            return False

        matching_count = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_count += 1
        
        # Reconstruct matching
        matched_edges = set()
        for u in range(n_left):
            if pair_u[u] != -1:
                matched_edges.add((left_nodes[u], right_nodes[pair_u[u]]))
        
        return matched_edges

    def decompose_with_matching(edges, adj, time_limit=60.0):
        """
        Decompose edges into paths using Maximum Matching on the line graph bipartite representation.
        """
        if not edges:
            return []

        # Create left and right nodes for each directed edge
        # Left node: (u, v) representing edge u->v
        # Right node: (u, v) representing edge u->v
        # Edge in bipartite graph: (u, v)_L -- (w, x)_R if v == w
        # This means edge (u,v) is followed by (v,x) in the path.
        
        left_nodes = []
        right_nodes = []
        # Map from directed edge to node id
        # We can just use the tuple (u, v) as the id
        
        directed_edges = []
        for u, v in edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))
            
        left_nodes = list(directed_edges)
        right_nodes = list(directed_edges)
        
        # Build bipartite graph edges
        # For each directed edge (u, v), it can be followed by any directed edge (v, w)
        # where (v, w) is in the graph.
        # Note: We must avoid using the same undirected edge twice.
        # The matching will select a set of directed edges.
        # If (u, v) is matched to (v, w), it means u-v-w is a segment.
        # The set of matched directed edges forms disjoint paths and cycles.
        # Since we are matching left to right, and each node has capacity 1,
        # each directed edge is used at most once as a "start" and at most once as an "end".
        # This ensures that in the resulting structure, each undirected edge is used at most once?
        # Let's check:
        # If we use (u, v) in the matching, it means the undirected edge {u, v} is part of a path segment.
        # Can {u, v} be used again as (v, u)?
        # If (u, v)_L is matched to (v, w)_R, and (v, u)_L is matched to (u, z)_R.
        # This would imply the path goes ... -> u -> v -> w ... and ... -> v -> u -> z ...
        # This uses the edge {u, v} twice.
        # To prevent this, we should not allow both (u, v) and (v, u) to be part of the matching
        # in a way that consumes the edge.
        # Actually, the standard reduction for edge-disjoint paths uses a different construction or post-processing.
        # A simpler check: if the matching uses both (u, v) and (v, u), it's invalid.
        # We can filter out such matchings or prevent them.
        
        # To be safe and simple, we will generate the matching, then check for edge conflicts.
        # If a conflict exists, we remove one of the conflicting arcs from the matching.
        
        # Build adjacency for bipartite graph
        # For efficiency, we group right nodes by their start vertex.
        # right_nodes_by_start[v] = list of (w, x) where x == v? No, (v, w) starts at v.
        # Wait, my definition:
        # Left: (u, v) means we are at v, having come from u.
        # Right: (w, x) means we are at x, having come from w.
        # Connection: (u, v)_L connects to (v, w)_R.
        # So for a left node (u, v), we connect to all right nodes (v, w) in the graph.
        
        right_by_start = defaultdict(list)
        for w, x in right_nodes:
            right_by_start[w].append((w, x))
            
        bip_edges = []
        for u, v in left_nodes:
            # Connect (u, v)_L to all (v, w)_R
            for v, w in right_by_start[v]:
                # Avoid self-loop in bipartite graph? (u, v) to (v, u) is allowed in bipartite graph,
                # but corresponds to backtracking in the original graph, which is not a simple path.
                # We allow it in the graph, but will filter later.
                bip_edges.append(((u, v), (v, w)))
                
        # Run Hopcroft-Karp
        matched = hopcroft_karp(left_nodes, right_nodes, bip_edges)
        
        # Reconstruct paths
        # matched contains pairs ((u, v), (v, w))
        # This means (u, v) is followed by (v, w).
        # We build a next pointer map: next[(u, v)] = (v, w)
        next_arc = {}
        for l, r in matched:
            next_arc[l] = r
            
        # Identify start nodes: left nodes that are not in the range of next_arc?
        # No, start nodes are left nodes that are not the target of any match?
        # Wait, the match is L -> R.
        # If (u, v) is matched to (v, w), then (u, v) is NOT a start of a new segment?
        # Actually, (u, v) is the PREDECESSOR of (v, w).
        # A node (a, b) is a START if it is not the target of any match.
        # i.e., (a, b) is not in {r for (l, r) in matched}.
        
        targets = {r for l, r in matched}
        starts = [l for l in left_nodes if l not in targets]
        
        paths = []
        visited_arcs = set()
        
        for start in starts:
            if start in visited_arcs:
                continue
                
            path = []
            current = start
            is_valid = True
            
            while current is not None:
                if current in visited_arcs:
                    # Cycle detected or conflict
                    is_valid = False
                    break
                visited_arcs.add(current)
                u, v = current
                path.append(u)
                path.append(v)
                
                # Move to next
                if current in next_arc:
                    current = next_arc[current]
                else:
                    current = None
                    
            if is_valid and len(path) > 1:
                # Check for simple path (no repeated vertices)
                if len(set(path)) == len(path):
                    paths.append(path)
                else:
                    # Split into simple paths
                    # This is a heuristic split
                    seen = set()
                    sub_path = []
                    for vertex in path:
                        if vertex in seen:
                            if len(sub_path) > 1:
                                paths.append(sub_path)
                            sub_path = []
                        seen.add(vertex)
                        sub_path.append(vertex)
                    if len(sub_path) > 1:
                        paths.append(sub_path)
                    
        # Check for unused edges
        used_edges = set()
        for p in paths:
            for i in range(len(p) - 1):
                used_edges.add(tuple(sorted((p[i], p[i+1]))))
                
        for e in edges:
            if tuple(sorted(e)) not in used_edges:
                paths.append(list(e))
                
        return paths

    def merge_paths(paths, edges, time_limit=20.0):
        """Post-processing: merge paths that share endpoints."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        changed = True
        iterations = 0
        max_iterations = 50

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() - start_time > time_limit:
                break

            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))
                endpoint_map[path[-1]].append((i, 1))

            merged = False
            new_paths = list(paths)

            for vertex in list(endpoint_map.keys()):
                if time.time() - start_time > time_limit:
                    break
                entries = endpoint_map[vertex]
                if len(entries) < 2:
                    continue

                for a in range(len(entries)):
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j:
                            continue
                        if i >= len(new_paths) or j >= len(new_paths):
                            continue

                        pi = new_paths[i]
                        pj = new_paths[j]

                        merged_path = None
                        if ei == 1 and ej == 0 and pi[-1] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[1:]
                        elif ei == 0 and ej == 1 and pi[0] == vertex and pj[-1] == vertex:
                            merged_path = pj + pi[1:]
                        elif ei == 1 and ej == 1 and pi[-1] == vertex and pj[-1] == vertex:
                            merged_path = pi + pj[::-1][1:]
                        elif ei == 0 and ej == 0 and pi[0] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[::-1][1:]

                        if merged_path is None:
                            continue

                        if len(merged_path) != len(set(merged_path)):
                            continue

                        valid = True
                        for k in range(len(merged_path) - 1):
                            u, v = merged_path[k], merged_path[k + 1]
                            if tuple(sorted((u, v))) not in edge_set:
                                valid = False
                                break

                        if not valid:
                            continue

                        remove_idx = sorted([i, j], reverse=True)
                        for idx in remove_idx:
                            new_paths.pop(idx)
                        new_paths.append(merged_path)
                        merged = True
                        changed = True
                        break
                    if merged:
                        break
                if merged:
                    break

            paths = new_paths

        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() - start_time > deadline - 10:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        adj = get_adjacency(edges)

        # Use Matching-based decomposition
        matching_paths = decompose_with_matching(edges, adj, time_limit=deadline - 20)
        
        # Merge pass
        final_paths = merge_paths(matching_paths, edges, time_limit=deadline - 10)
        
        results.append(final_paths)

    return results
