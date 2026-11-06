"""Rewritten program: ILP-based maximum edge-disjoint cycle packing."""
import itertools
from random import Random
import sys

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_instance(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        edge_list = list(edges)
        m = len(edge_list)
        
        # Map edge to index
        edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edge_list)}
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all simple cycles up to length 6
        # This is a heuristic to keep the ILP tractable
        # We look for cycles of length 3, 4, 5, 6
        
        cycles = []
        seen_cycles = set()
        
        def find_cycles_from(start, max_len):
            """Find all simple cycles starting and ending at 'start' up to max_len."""
            for end in sorted(adj[start]):
                if end < start:
                    continue
                # Path from start to end, then edge back to start
                # We need paths of length (max_len - 1) from start to end
                # that don't use the edge (start, end) directly unless it's the closing edge
                
                # Actually, let's use a DFS to find all simple paths from start to end
                # of length k, where k+1 = cycle length
                # The cycle is start -> ... -> end -> start
                
                # We'll find paths of various lengths
                for path_len in range(1, max_len):  # path_len is number of edges in path from start to end
                    if path_len == 1:
                        # Direct edge start-end, cycle length 2 (not valid)
                        continue
                    # Find all simple paths from start to end of exactly path_len edges
                    # that don't include the direct edge (start, end) in intermediate steps
                    stack = [(start, [start])]
                    found = []
                    while stack:
                        node, path = stack.pop()
                        if len(path) - 1 == path_len:
                            if node == end:
                                found.append(path)
                            continue
                        if len(path) - 1 > path_len:
                            continue
                        for nb in sorted(adj[node]):
                            if nb in path:
                                continue
                            # Don't go back to start unless we're closing the cycle
                            # But we're looking for paths to 'end', so we can go through start? No, simple path
                            if nb == start and len(path) > 1:
                                continue
                            stack.append((nb, path + [nb]))
                    
                    for path in found:
                        cycle_vertices = path + [start]
                        # Canonical form: rotate so smallest vertex is first, and direction is consistent
                        min_idx = cycle_vertices.index(min(cycle_vertices))
                        rotated = cycle_vertices[min_idx:] + cycle_vertices[:min_idx]
                        # Two directions
                        rev = rotated[::-1]
                        canonical = min(tuple(rotated), tuple(rev))
                        if canonical not in seen_cycles:
                            seen_cycles.add(canonical)
                            cycles.append(canonical)
        
        # Find cycles of length 3 to 6
        for start in range(n):
            find_cycles_from(start, 6)
        
        # If no cycles found, return all edges
        if not cycles:
            return [list(e) for e in edge_list]
        
        # Build ILP
        # Variables: x_c for each cycle c (1 if used), x_e for each edge e (1 if unused)
        # Minimize: sum(x_c) + sum(x_e)
        # Constraints: for each edge e, sum(x_c for c containing e) + x_e = 1
        
        num_cycles = len(cycles)
        
        # Precompute which cycles contain which edges
        cycle_edges = []
        for cyc in cycles:
            edges_in_cycle = set()
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                e = tuple(sorted((u, v)))
                edges_in_cycle.add(e)
            cycle_edges.append(edges_in_cycle)
        
        # Solve ILP using scipy.optimize.milp or a simple branch and bound
        # Since the problem size is small (n<=30, m<=~200), we can use a simple approach
        # Let's use scipy.optimize.milp if available, otherwise fall back to greedy
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            # Objective: minimize sum(x_c) + sum(x_e)
            # x_c in [0,1], x_e in [0,1]
            # Integer: x_c, x_e are binary
            
            c_obj = np.ones(num_cycles + m)
            
            # Constraints: for each edge e, sum(x_c for c containing e) + x_e = 1
            # A matrix of shape (m, num_cycles + m)
            A = np.zeros((m, num_cycles + m))
            for e_idx, e in enumerate(edge_list):
                A[e_idx, e_idx + num_cycles] = 1  # x_e term
                for c_idx, cyc_edges in enumerate(cycle_edges):
                    if e in cyc_edges:
                        A[e_idx, c_idx] = 1
            
            # Bounds
            bounds = Bounds(lb=np.zeros(num_cycles + m), ub=np.ones(num_cycles + m))
            
            # Linear constraints: A @ x = 1
            constraints = LinearConstraint(A, lb=np.ones(m), ub=np.ones(m))
            
            # Integrality: all variables are binary
            integrality = np.ones(num_cycles + m)
            
            res = milp(c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
            
            if res.status == 0:
                x = res.x
                result = []
                # Extract cycles
                for c_idx in range(num_cycles):
                    if x[c_idx] > 0.5:
                        result.append(list(cycles[c_idx]))
                # Extract unused edges
                for e_idx in range(m):
                    if x[e_idx + num_cycles] > 0.5:
                        result.append(list(edge_list[e_idx]))
                return result
            else:
                # Fallback to greedy
                pass
        
        except Exception:
            pass
        
        # Fallback: greedy cycle packing
        # Greedily pick cycles that cover the most uncovered edges
        remaining_edges = set(edge_list)
        result = []
        
        # Sort cycles by length (prefer longer cycles? or shorter?)
        # Longer cycles cover more edges, so prefer longer
        cycles_sorted = sorted(cycles, key=len, reverse=True)
        
        used_edges = set()
        for cyc in cycles_sorted:
            cyc_edge_set = set()
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                e = tuple(sorted((u, v)))
                cyc_edge_set.add(e)
            
            if cyc_edge_set.issubset(remaining_edges):
                result.append(list(cyc))
                remaining_edges -= cyc_edge_set
        
        # Add remaining edges
        for e in remaining_edges:
            result.append(list(e))
        
        return result
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = instance(*spec)
        pieces = solve_instance(edges)
        results.append(pieces)
    
    return results
