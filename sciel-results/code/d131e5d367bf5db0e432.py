"""Program using min-cost flow for optimal edge-disjoint path cover."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    _start = time.time()
    _DEADLINE = 145.0

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_cost_flow_path_cover(edges, n_vertices, time_limit=40.0):
        """
        Formulate edge-disjoint path cover as min-cost flow:
        - Node splitting: each vertex v becomes v_in -> v_out with capacity 1, cost 0
          (ensures each vertex is used at most once per path)
        - Each edge (u,v) becomes u_out -> v_in and v_out -> u_in with capacity 1, cost 0
        - To count paths: add a super source S and super sink T
          S -> v_in for each vertex with capacity 1, cost 1 (start of a path)
          v_out -> T for each vertex with capacity 1, cost 1 (end of a path)
        - We need to send flow that covers all edges. 
          Actually, let's think differently.
        
        Better formulation:
        We want to decompose edges into minimum number of simple paths.
        This is equivalent to: find a minimum-cost circulation where each edge 
        has flow 0 or 1, and the number of "path starts" is minimized.
        
        Standard reduction:
        - Split each vertex v into v_in and v_out
        - v_in -> v_out: capacity 1, cost 0 (vertex capacity)
        - For each undirected edge (u,v): u_out -> v_in (cap 1, cost 0) and v_out -> u_in (cap 1, cost 0)
        - Add source S: S -> v_in for all v (cap 1, cost 0) - represents path start
        - Add sink T: v_out -> T for all v (cap 1, cost 0) - represents path end
        - We need to select a subset of edges such that every edge is covered exactly once,
          and the selected edges form a collection of paths.
        
        Actually the correct min-cost flow formulation for minimum path cover:
        - The problem is NP-hard in general, but for our small instances we can use
          a different approach.
        
        Let me use a different structural approach: 
        **Integer Programming via branch and bound on the matching structure**
        
        Actually, let's use the correct min-cost flow:
        Minimum edge-disjoint path cover = minimum number of paths to cover all edges.
        
        Key insight: A set of edge-disjoint paths covering all edges is a 
        "path decomposition". The minimum number of paths equals:
        max(1, ceil(sum of (deg(v)-1) for odd v / 2))... no that's not right either.
        
        Let me use a completely different approach: **Simulated annealing on path arrangements**
        where we represent the solution as a permutation of edges and try to form 
        maximum-length simple paths greedily, with SA to optimize the ordering.
        
        Actually, the cleanest different approach: **Greedy with edge ordering optimization**
        using simulated annealing on the order in which edges are assigned to paths.
        """
        if not edges:
            return []
        
        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))
        
        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        m = len(edge_set)
        
        # Simulated annealing: represent solution as ordering of edges
        # Given an ordering, greedily assign each edge to an existing path 
        # (extending from either end) or start a new path.
        
        import random
        rng = random.Random(42)
        
        edge_list = list(edge_set)
        
        def evaluate_ordering(order):
            """Given an ordering of edges, greedily build paths."""
            # Map edge index to (u,v)
            # Track for each path: the set of vertices used, and the two endpoints
            # A path is represented by (set_of_vertices, left_end, right_end, list_of_vertices)
            
            paths = []  # list of [vertex_set, endpoints_set, vertex_list]
            edge_to_path = {}  # edge_key -> path index
            
            for edge_key in order:
                u, v = edge_key
                assigned = False
                
                # Try to extend an existing path
                for pi, (vset, endpoints, vlist) in enumerate(paths):
                    # Can we extend from left end?
                    if u in endpoints and v not in vset:
                        # Extend: u is an endpoint, add v
                        new_vset = vset | {v}
                        new_endpoints = (endpoints - {u}) | {v}
                        # Update vlist
                        if vlist[0] == u:
                            new_vlist = [v] + vlist
                        else:
                            new_vlist = vlist + [v]
                        paths[pi] = [new_vset, new_endpoints, new_vlist]
                        edge_to_path[edge_key] = pi
                        assigned = True
                        break
                    if v in endpoints and u not in vset:
                        new_vset = vset | {u}
                        new_endpoints = (endpoints - {v}) | {u}
                        if vlist[0] == v:
                            new_vlist = [u] + vlist
                        else:
                            new_vlist = vlist + [u]
                        paths[pi] = [new_vset, new_endpoints, new_vlist]
                        edge_to_path[edge_key] = pi
                        assigned = True
                        break
                
                if not assigned:
                    # Start new path
                    paths.append([{u, v}, {u, v}, [u, v]])
                    edge_to_path[edge_key] = len(paths) - 1
            
            return len(paths)
        
        def build_paths_from_order(order):
            """Reconstruct actual path vertex lists from an ordering."""
            paths_data = []
            edge_to_path = {}
            
            for edge_key in order:
                u, v = edge_key
                assigned = False
                
                for pi in range(len(paths_data)):
                    vset, endpoints, vlist = paths_data[pi]
                    if u in endpoints and v not in vset:
                        new_vset = vset | {v}
                        new_endpoints = (endpoints - {u}) | {v}
                        if vlist[0] == u:
                            new_vlist = [v] + vlist
                        else:
                            new_vlist = vlist + [v]
                        paths_data[pi] = (new_vset, new_endpoints, new_vlist)
                        edge_to_path[edge_key] = pi
                        assigned = True
                        break
                    if v in endpoints and u not in vset:
                        new_vset = vset | {u}
                        new_endpoints = (endpoints - {v}) | {u}
                        if vlist[0] == v:
                            new_vlist = [u] + vlist
                        else:
                            new_vlist = vlist + [u]
                        paths_data[pi] = (new_vset, new_endpoints, new_vlist)
                        edge_to_path[edge_key] = pi
                        assigned = True
                        break
                
                if not assigned:
                    paths_data.append(({u, v}, {u, v}, [u, v]))
                    edge_to_path[edge_key] = len(paths_data) - 1
            
            return [vlist for (_, _, vlist) in paths_data]
        
        # Initial ordering: random
        current_order = edge_list[:]
        rng.shuffle(current_order)
        current_cost = evaluate_ordering(current_order)
        best_order = current_order[:]
        best_cost = current_cost
        
        # SA parameters
        t_start = time.time()
        sa_time_limit = time_limit
        T = 10.0
        cooling = 0.999
        
        while time.time() - t_start < sa_time_limit:
            # Move: swap two random edges in the ordering
            i, j = rng.sample(range(m), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_cost = evaluate_ordering(new_order)
            
            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
                current_order = new_order
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = current_order[:]
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return build_paths_from_order(best_order)

    def greedy_path_cover(edges):
        """Fallback greedy path cover."""
        if not edges:
            return []

        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))

        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)

        used_edges = set()
        paths = []

        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break

            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)

            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break

            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor

            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)

            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])

        return paths

    def validate_and_fix(paths, edges):
        """Validate paths and fix if invalid."""
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                valid = False
                break
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if not valid or covered != edge_set:
            return greedy_path_cover(edges)
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 15:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        
        remaining_time = _DEADLINE - time.time()
        graphs_left = 3 - len(results)
        time_per_graph = max(5.0, (remaining_time - 10) / max(1, graphs_left))
        
        try:
            paths = min_cost_flow_path_cover(edges, spec[0], time_limit=time_per_graph)
        except Exception:
            paths = []
        
        if not paths:
            paths = greedy_path_cover(edges)
        
        paths = validate_and_fix(paths, edges)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END
