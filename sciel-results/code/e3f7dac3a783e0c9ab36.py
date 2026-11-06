"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
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

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            # Check simple path
            if len(set(path)) != len(path):
                return False
            # Check edges exist
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges covered
        if used_edges != all_edges:
            return False
        
        return True

    def greedy_trail_decomposition(edges, adj, n, max_time=40):
        """
        Decompose graph into edge-disjoint simple paths using greedy longest-path search.
        Repeatedly find a long simple path in the remaining graph and remove its edges.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Build adjacency for remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        while remaining:
            if not remaining:
                break
            
            # Find vertices with highest degree in remaining graph
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if neigh}
            if not degs:
                break
            
            # Try multiple starting points for diversity
            candidates = sorted(degs.keys(), key=lambda v: -degs[v])[:8]
            
            best_path = None
            best_len = 0
            
            for cand_start in candidates:
                if not remaining:
                    break
                
                # Greedy walk to find a long simple path
                path = [cand_start]
                visited_set = {cand_start}
                current = cand_start
                
                while True:
                    extended = False
                    # Sort neighbors by remaining degree (descending)
                    neighbors = list(rem_adj[current])
                    neighbors.sort(key=lambda nb: -len(rem_adj[nb]))
                    
                    for nb in neighbors:
                        e = tuple(sorted((current, nb)))
                        if e in remaining and nb not in visited_set:
                            path.append(nb)
                            visited_set.add(nb)
                            current = nb
                            extended = True
                            break
                    
                    if not extended:
                        break
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
            
            if best_path is None or len(best_path) < 2:
                # Fallback: pick any edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
                continue
            
            # Add the path and remove its edges
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                if e in remaining:
                    remaining.remove(e)
                    rem_adj[e[0]].discard(e[1])
                    rem_adj[e[1]].discard(e[0])
        
        return paths

    def merge_paths(paths, edges, adj, n, max_iterations=100):
        """Try to merge pairs of paths to reduce total count."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        for _ in range(max_iterations):
            merged = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    p1, p2 = current[i], current[j]
                    
                    # Try 4 ways to merge
                    for a, b, orientation in [
                        (p1[-1], p2[0], 'normal'),
                        (p1[0], p2[-1], 'normal'),
                        (p1[-1], p2[-1], 'reverse'),
                        (p1[0], p2[0], 'reverse')
                    ]:
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        
                        # Check no vertex overlap
                        if set(p1) & set(p2):
                            continue
                        
                        # Build merged path
                        if orientation == 'normal':
                            if a == p1[-1] and b == p2[0]:
                                merged_path = p1 + p2
                            elif a == p1[0] and b == p2[-1]:
                                merged_path = p2 + p1
                        else:  # reverse
                            if a == p1[-1] and b == p2[-1]:
                                merged_path = p1 + p2[::-1]
                            elif a == p1[0] and b == p2[0]:
                                merged_path = p1[::-1] + p2
                        
                        # Verify simplicity
                        if len(set(merged_path)) != len(merged_path):
                            continue
                        
                        # Verify all edges exist
                        valid = True
                        for k in range(len(merged_path) - 1):
                            e2 = tuple(sorted((merged_path[k], merged_path[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        
                        if valid:
                            # Create new list without i and j
                            new_current = []
                            for idx in range(len(current)):
                                if idx != i and idx != j:
                                    new_current.append(current[idx])
                            new_current.append(merged_path)
                            
                            # Verify full decomposition
                            if verify_decomposition(new_current, edges, n):
                                current = new_current
                                merged = True
                                break
                    
                    if merged:
                        break
                
                if merged:
                    break
            
            if not merged:
                break
        
        return current

    def solve_milp_path_decomposition(edges, n, time_limit=30):
        """
        Use MILP to find minimum number of edge-disjoint simple paths covering all edges.
        
        Formulation:
        - For each directed arc (u,v) in the line graph, variable x_uv in {0,1}
        - For each vertex v, flow conservation: sum_in - sum_out = b_v where b_v in {-1,0,1}
        - Minimize sum of b_v (number of path endpoints / 2, but we count paths)
        
        Actually, let's use a cleaner formulation:
        - Each original edge e=(u,v) must be used exactly once (as either (u,v) or (v,u))
        - For each vertex, the number of path starts/ends is determined by degree parity
        - Minimize total number of paths = (number of vertices with odd "path endpoint" count) / 2
        
        Better approach: Model as minimum path cover in directed sense.
        For each undirected edge, choose a direction. Then find minimum number of paths
        such that each edge is in exactly one path.
        
        Variables:
        - x_e for each directed arc: 1 if arc is used
        - y_v for each vertex: 1 if vertex is a path endpoint (start or end)
        
        Constraints:
        - For each undirected edge {u,v}: x_uv + x_vu = 1
        - For each vertex v: sum_out(v) - sum_in(v) = y_v - (1 - y_v) = 2*y_v - 1
          Wait, that's not right. Let me think again.
          
        Actually, for a path decomposition:
        - Each path has exactly 2 endpoints (start and end)
        - For vertex v: out_degree(v) - in_degree(v) = 1 if v is a start, -1 if v is an end, 0 otherwise
        - Number of paths = (number of starts + number of ends) / 2 = number of starts (since each path has 1 start)
        
        So: minimize sum of start_v where start_v in {0,1}
        Subject to: for each v, out(v) - in(v) = start_v - end_v, and start_v + end_v <= 1
        But actually start_v and end_v can both be 0 or one can be 1.
        
        Let me use: for each vertex v, define net_v = out(v) - in(v)
        Then net_v must be in {-1, 0, 1}
        Number of paths = sum of max(0, net_v) = number of vertices with net_v = 1
        
        To linearize: introduce s_v, t_v in {0,1} where s_v = 1 if v is a start, t_v = 1 if v is an end
        net_v = s_v - t_v
        Minimize sum(s_v)
        
        For each directed arc (u,v): x_uv in {0,1}
        For each undirected edge {u,v}: x_uv + x_vu = 1
        For each vertex v: sum_u x_uv - sum_u x_vu = s_v - t_v
        s_v, t_v in {0,1}, and s_v * t_v = 0 (can't be both) -> s_v + t_v <= 1
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
        except ImportError:
            return None
        
        # Build directed arcs
        arcs = []
        arc_index = {}
        for u, v in edges:
            arcs.append((u, v))
            arcs.append((v, u))
            arc_index[(u, v)] = len(arcs) - 2
            arc_index[(v, u)] = len(arcs) - 1
        
        m = len(arcs)  # number of arc variables
        # Variables: x[0..m-1] for arcs, s[n], t[n] for vertices
        # Total variables: m + 2*n
        total_vars = m + 2 * n
        
        # Objective: minimize sum of s_v
        c = np.zeros(total_vars)
        for v in range(n):
            c[m + v] = 1  # s_v
        
        # Constraints:
        # 1. For each undirected edge {u,v}: x_uv + x_vu = 1
        # 2. For each vertex v: sum_out - sum_in = s_v - t_v
        # 3. s_v + t_v <= 1 for all v
        
        # Build constraint matrix
        A_rows = []
        lb_list = []
        ub_list = []
        
        # Constraint 1: edge constraints
        for u, v in edges:
            row = np.zeros(total_vars)
            row[arc_index[(u, v)]] = 1
            row[arc_index[(v, u)]] = 1
            A_rows.append(row)
            lb_list.append(1)
            ub_list.append(1)
        
        # Constraint 2: flow conservation
        # For each vertex v: sum_{u} x_{uv} - sum_{u} x_{vu} - s_v + t_v = 0
        for v in range(n):
            row = np.zeros(total_vars)
            for u, w in edges:
                if u == v:
                    row[arc_index[(v, w)]] = 1  # outgoing
                    row[arc_index[(w, v)]] = -1  # incoming
                elif w == v:
                    row[arc_index[(v, u)]] = 1  # outgoing
                    row[arc_index[(u, v)]] = -1  # incoming
            row[m + v] = -1  # -s_v
            row[m + n + v] = 1  # +t_v
            A_rows.append(row)
            lb_list.append(0)
            ub_list.append(0)
        
        # Constraint 3: s_v + t_v <= 1
        for v in range(n):
            row = np.zeros(total_vars)
            row[m + v] = 1
            row[m + n + v] = 1
            A_rows.append(row)
            lb_list.append(0)
            ub_list.append(1)
        
        if not A_rows:
            return None
        
        A = np.array(A_rows)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        # Bounds: all variables in [0, 1]
        bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
        
        # Integrality: all variables are integer
        integrality = np.ones(total_vars)
        
        constraints = LinearConstraint(A, lb, ub)
        
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_limit, "presolve": True}
            )
        except Exception:
            return None
        
        if not result.success:
            return None
        
        x = result.x
        # Extract path count
        num_paths = int(round(x[m:m+n].sum()))
        
        # Now reconstruct paths from the solution
        # x[m+v] = s_v (1 if v is a start)
        # x[m+n+v] = t_v (1 if v is an end)
        
        # Build directed graph from solution
        directed_edges = set()
        for i, (u, v) in enumerate(arcs):
            if x[i] > 0.5:
                directed_edges.add((u, v))
        
        # Find paths by following directed edges from starts
        adj_out = defaultdict(list)
        for u, v in directed_edges:
            adj_out[u].append(v)
        
        starts = [v for v in range(n) if x[m+v] > 0.5]
        paths = []
        used_arcs = set()
        
        for start in starts:
            path = [start]
            current = start
            while True:
                # Find next vertex
                next_v = None
                for nb in adj_out[current]:
                    if (current, nb) not in used_arcs:
                        next_v = nb
                        break
                if next_v is None:
                    break
                used_arcs.add((current, next_v))
                path.append(next_v)
                current = next_v
            paths.append(path)
        
        # Verify and return
        if verify_decomposition(paths, edges, n):
            return paths
        
        return None

    def solve_with_ilp_refinement(paths, edges, n, time_limit=20):
        """
        Given an initial path decomposition, try to improve it using local search / ILP.
        For now, just try merging more aggressively.
        """
        return paths

    def reduce_paths_by_splitting(paths, edges, n):
        """
        If a path is too long, we might be able to split it differently.
        This is complex; skip for now.
        """
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # Try MILP first for smaller instances
        if n <= 30 and time.time() - start_time < deadline - 40:
            milp_result = solve_milp_path_decomposition(edges, n, time_limit=25)
            if milp_result is not None:
                paths = milp_result
                results.append(paths)
                continue
        
        # Get initial decomposition via greedy
        paths = greedy_trail_decomposition(edges, adj, n)
        
        # Verify it's valid
        if not verify_decomposition(paths, edges, n):
            # Fallback: each edge as a path
            paths = [list(e) for e in edges]
        
        # Try to merge paths to reduce count
        if len(paths) > 1:
            merged = merge_paths(paths, edges, adj, n, max_iterations=50)
            if verify_decomposition(merged, edges, n):
                paths = merged
        
        # Try ILP refinement if time permits
        if time.time() - start_time < deadline - 15 and len(paths) > 5:
            refined = solve_with_ilp_refinement(paths, edges, n, time_limit=10)
            if verify_decomposition(refined, edges, n):
                paths = refined
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 10:
            # Run out of time, use fallback for remaining
            for idx in range(len(results), 3):
                n2, p2, seed2 = specs[idx]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
