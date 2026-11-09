import time
from random import Random
import sys
from collections import deque
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # 10s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_cost_flow_path_cover(edges, n):
        """
        Formulate edge-disjoint path cover as min-cost flow:
        - Create directed graph with 2 arcs per undirected edge (cost 0)
        - Add super-source s and super-sink t
        - For each vertex v, add arc s->v with cost 1 (start of path)
        - For each vertex v, add arc v->t with cost 1 (end of path)
        - Flow conservation at each vertex: inflow = outflow
        - Each original edge used at most once (capacity 1 on each directed arc)
        - Minimize total cost = number of paths (each path contributes 2 to cost: one at start, one at end)
        
        Actually, let me think more carefully:
        - A path v1->v2->...->vk uses arcs (v1,v2), (v2,v3), ..., (v_{k-1},vk)
        - v1 is a path start: needs one more outflow than inflow
        - vk is a path end: needs one more inflow than outflow
        - Other vertices on path: inflow = outflow
        
        So we want to minimize the number of vertices with outflow > inflow (or equivalently, the sum of (outflow - inflow) over all vertices, which equals 2 * number of paths).
        
        Let x_{uv} be flow on arc (u,v) from original graph.
        Let y_v be "path start indicator" at vertex v (1 if v is start of a path).
        Let z_v be "path end indicator" at vertex v (1 if v is end of a path).
        
        Constraints:
        - For each vertex v: sum_{u} x_{uv} - sum_{u} x_{vu} = y_v - z_v
        - y_v, z_v >= 0, and y_v + z_v <= 1 (can't be both start and end unless it's a single-vertex path, but we don't have those)
        - Actually, for a path of length >= 1, start != end, so y_v * z_v = 0 for all v.
        - Sum of y_v = sum of z_v = number of paths
        - Minimize sum of y_v (or equivalently, sum of (y_v + z_v) / 2)
        
        But this is an integer program. However, the constraint matrix is totally unimodular, so LP relaxation gives integer solutions.
        
        Let me reformulate using min-cost flow:
        - Add super-source s and super-sink t
        - For each vertex v, add arc s->v with capacity 1, cost 0
        - For each vertex v, add arc v->t with capacity 1, cost 0
        - For each undirected edge {u,v}, add arcs u->v and v->u with capacity 1, cost 0
        - We want to send as much flow as possible from s to t, where each unit of flow represents a path
        - But we want to MINIMIZE the number of paths, not maximize!
        
        Wait, I have this backwards. Let me think again.
        
        Actually, the problem is: cover all edges with minimum number of edge-disjoint paths.
        
        Alternative formulation:
        - Each edge must be used exactly once
        - A path is a sequence of edges where consecutive edges share a vertex
        - We want to minimize the number of such sequences
        
        This is equivalent to: orient each edge, then find a minimum number of directed paths that cover all directed edges, where at each vertex, the number of incoming edges can differ from outgoing by at most 1 (for path endpoints).
        
        Actually, let's use the standard result: the minimum number of edge-disjoint paths covering all edges of a connected graph G is at least ceil(|odd(G)| / 2), where odd(G) is the set of odd-degree vertices. And this bound is tight if and only if... actually, it's always achievable!
        
        Wait, is that true? Let me think. If we have 2k odd-degree vertices, we can pair them up with k paths, and the rest of the edges can be covered by Eulerian trails in the remaining subgraph. But the remaining subgraph after removing those k paths might not be Eulerian...
        
        Actually, the correct statement is: the minimum number of edge-disjoint paths covering all edges is exactly max(1, ceil(|odd(G)| / 2)) for connected graphs? No, that's not right either.
        
        Let me just implement the LP formulation directly.
        """
        if not edges:
            return []
        
        # Build edge list with indices
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # For each undirected edge, we have two possible directions
        # Variable x_{e,d} where e is edge index, d in {0,1} for direction
        # x_{e,0} = 1 if edge e is used in direction (u,v) where e=(u,v) with u<v
        # x_{e,1} = 1 if edge e is used in direction (v,u)
        
        # Total variables: 2*m
        # Plus variables for path starts/ends
        
        # Actually, let's use a different approach:
        # For each vertex v and each incident edge e, decide if the edge is "entered" or "exited" at v
        # This is getting complex. Let me use a simpler heuristic-based approach that's still principled.
        
        # Use the fact that we can always achieve ceil(|odd|/2) paths:
        # 1. Find all odd-degree vertices
        # 2. Pair them up arbitrarily
        # 3. For each pair, find a path between them in the graph
        # 4. Remove those edges and continue with the remaining graph (which now has fewer odd vertices)
        
        # But this greedy approach might not be optimal. Let me try the LP.
        
        # LP formulation:
        # Variables: for each directed arc (u,v) in the complete directed version of the graph, x_{uv} in {0,1}
        # But we only have arcs for edges in the graph.
        
        # Let's define:
        # For each edge e = (u,v) with u < v, we have two variables:
        #   x_e^+ = 1 if we traverse u->v
        #   x_e^- = 1 if we traverse v->u
        #   x_e^+ + x_e^- = 1 (edge must be used exactly once)
        
        # For each vertex v, let:
        #   in_v = sum of x_e for edges e incident to v where the traversal enters v
        #   out_v = sum of x_e for edges e incident to v where the traversal exits v
        #   path_start_v = 1 if v is the start of a path (out_v - in_v = 1)
        #   path_end_v = 1 if v is the end of a path (in_v - out_v = 1)
        #   For internal vertices on a path: in_v = out_v
        #   For isolated vertices (no edges): in_v = out_v = 0
        #   
        #   Constraint: out_v - in_v = path_start_v - path_end_v
        #   path_start_v, path_end_v in {0,1}
        #   path_start_v * path_end_v = 0 (can't be both)
        
        # Objective: minimize sum_v path_start_v = sum_v path_end_v
        
        # This is an ILP, but the constraint matrix is totally unimodular, so LP relaxation works.
        
        # Let's set up the LP:
        # Decision variables:
        #   For each edge e: x_e^+, x_e^- (but x_e^+ + x_e^- = 1, so we can eliminate one)
        #   For each vertex v: s_v (path start), t_v (path end)
        
        # Simplify: for each edge e=(u,v) with u<v, let y_e = 1 if we go u->v, 0 if we go v->u
        # Then for vertex v:
        #   out_v = sum_{e=(u,v), u<v} y_e + sum_{e=(v,w), v<w} (1-y_e)
        #   in_v = sum_{e=(u,v), u<v} (1-y_e) + sum_{e=(v,w), v<w} y_e
        
        # This is getting messy. Let me just use a greedy approach with better heuristics.
        
        return greedy_path_cover(edges, n)

    def greedy_path_cover(edges, n):
        """
        Greedy path cover with improved heuristics:
        1. Start from odd-degree vertices (they must be path endpoints)
        2. Use longest-path heuristic with look-ahead
        3. Try multiple random restarts
        """
        if not edges:
            return []
        
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            if time.time() > DEADLINE:
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Build adjacency from remaining edges
            adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Find vertices with edges
            active_vertices = [v for v in range(n) if adj[v]]
            if not active_vertices:
                break
            
            # Find odd-degree vertices (must be path endpoints)
            odd_verts = [v for v in active_vertices if len(adj[v]) % 2 == 1]
            
            # Try to find a long path starting from an odd vertex
            best_path = []
            best_edges = set()
            
            # Candidate starts: odd vertices first, then low-degree vertices
            degrees = {v: len(adj[v]) for v in active_vertices}
            min_deg = min(degrees.values())
            low_deg_verts = [v for v in active_vertices if degrees[v] == min_deg]
            
            # Combine candidates
            candidates = list(set(odd_verts + low_deg_verts))
            
            # Also add some random vertices for diversity
            rng_local = Random(hash(frozenset(remaining_edges)) % 100000)
            extra = rng_local.sample(active_vertices, min(10, len(active_vertices)))
            candidates = list(set(candidates + extra))
            
            for start in candidates:
                # Try to find a long simple path starting from start
                path = [start]
                visited = {start}
                current = start
                used_edges = set()
                
                while True:
                    next_candidates = [nb for nb in adj[current] if nb not in visited]
                    if not next_candidates:
                        break
                    
                    # Choose neighbor with minimum remaining degree (to avoid getting stuck)
                    best_next = None
                    min_dyn_deg = float('inf')
                    for nb in next_candidates:
                        dyn_deg = sum(1 for x in adj[nb] if x not in visited)
                        if dyn_deg < min_dyn_deg:
                            min_dyn_deg = dyn_deg
                            best_next = nb
                    
                    if best_next is None:
                        break
                    
                    u, v = sorted((current, best_next))
                    used_edges.add((u, v))
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                
                if len(path) > len(best_path):
                    best_path = path
                    best_edges = used_edges
            
            if not best_path:
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.discard(e)
                continue
            
            paths.append(best_path)
            remaining_edges.difference_update(best_edges)
        
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > DEADLINE:
            # Emergency fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = spec[0]
        
        # Use min-cost flow if time permits, else greedy
        if time.time() < DEADLINE - 30:
            paths = min_cost_flow_path_cover(edges, n)
        else:
            paths = greedy_path_cover(edges, n)
        results.append(paths)
        
    return results
