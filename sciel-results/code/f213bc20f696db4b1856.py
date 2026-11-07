"""Improved program for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 15}

def construct():
    t0 = time.time()
    deadline = t0 + 85  # 5s safety margin before 90s kill
    n = P['n']
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    edge_set_all = set(all_edges)
    
    # Precompute adjacency info for fast degree checking
    # We'll use a bitmask or adjacency matrix approach for speed
    
    def build_adj(edges):
        """Build adjacency matrix and degree list from edge list."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def is_valid_edges(edge_list):
        """Check if the graph defined by edge_list satisfies the constraint.
        
        No simple 3-edge path s-u-v-t with s != t should have deg(s) == deg(t).
        """
        if not edge_list:
            return True
        
        adj, deg = build_adj(edge_list)
        
        # For every edge (u, v), consider all paths s-u-v-t where s != t, s != u, s != v, t != u, t != v
        # Actually, s is a neighbor of u (s != v), t is a neighbor of v (t != u), and s != t
        # Wait, the path is s-u-v-t, so edges are (s,u), (u,v), (v,t)
        # s is adjacent to u, s != v (since path is simple, s != u, s != v, t != u, t != v, s != t)
        # Actually for a simple path s-u-v-t: s, u, v, t are all distinct.
        
        # So for each edge (u,v) in edge_list:
        #   for each s in adj[u] where s != v:
        #     for each t in adj[v] where t != u and t != s:
        #       if deg[s] == deg[t]: return False
        
        for u, v in edge_list:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Start with a good initial solution: star + one extra edge
    # Star centered at 0, plus edge (1,2)
    initial_edges = [(0, v) for v in range(1, n)] + [(1, 2)]
    initial_edges_set = set(initial_edges)
    
    best_edges = list(initial_edges)
    best_score = len(best_edges)
    
    # Verify initial is valid
    if not is_valid_edges(best_edges):
        # Fallback to pure star
        best_edges = [(0, v) for v in range(1, n)]
        best_score = len(best_edges)
    
    # Local search: try adding one edge at a time
    # Also try swaps: remove one edge, add another
    
    def try_add(edge_list, edge_set, candidate_edge):
        """Try adding candidate_edge to edge_list. Return new edge list if valid, else None."""
        if candidate_edge in edge_set:
            return None
        new_list = edge_list + [candidate_edge]
        if is_valid_edges(new_list):
            return new_list
        return None
    
    def try_swap(edge_list, edge_set, remove_edge, add_edge):
        """Try removing remove_edge and adding add_edge. Return new edge list if valid, else None."""
        if remove_edge not in edge_set:
            return None
        if add_edge in edge_set:
            return None
        new_list = [e for e in edge_list if e != remove_edge] + [add_edge]
        if is_valid_edges(new_list):
            return new_list
        return None
    
    # Greedy improvement: repeatedly try to add an edge
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_edges = list(best_edges)
        current_edge_set = set(best_edges)
        for candidate in all_edges:
            if candidate in current_edge_set:
                continue
            result = try_add(current_edges, current_edge_set, candidate)
            if result is not None:
                best_edges = result
                best_score = len(best_edges)
                changed = True
                # Restart from scratch for speed
                break
    
    # Try swaps to improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = list(best_edges)
        current_edge_set = set(best_edges)
        
        for remove_e in list(current_edge_set):
            if time.time() > deadline:
                break
            for add_e in all_edges:
                if add_e in current_edge_set:
                    continue
                if add_e == remove_e:
                    continue
                result = try_swap(current_edges, current_edge_set, remove_e, add_e)
                if result is not None and len(result) > len(best_edges):
                    best_edges = result
                    best_score = len(best_edges)
                    improved = True
                    break
            if improved:
                break
    
    # Return edges as list of [u, v] with Python ints
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
