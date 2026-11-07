"""Constraint-satisfaction approach: group vertices by degree, assign edges between
groups such that no distance-3 path has endpoints of equal degree."""
import sys
from itertools import combinations, product
import random
from collections import defaultdict

P = {'n': 15}


def construct():
    n = P['n']

    # Precompute all simple 3-edge paths: (s, u, v, t) where s-u-v-t are consecutive
    # and s != t, s != v, u != t, and all edges (s,u),(u,v),(v,t) must exist.
    # We'll enumerate all ordered quadruples (s, u, v, t) with s != t, s != v, u != t,
    # and edges (s,u), (u,v), (v,t) present.
    
    all_pairs = list(combinations(range(n), 2))
    all_pairs_set = set(all_pairs)
    
    # Enumerate all possible 3-edge paths (s, u, v, t)
    # s-u, u-v, v-t must be edges, s != t
    # s, u, v, t are distinct? No, just s != t and the path is simple (no repeated vertices)
    # Actually "simple three-edge path" means no repeated vertices.
    # So s, u, v, t all distinct, and edges (s,u), (u,v), (v,t) exist.
    
    paths = []
    for s in range(n):
        for u in range(n):
            if s == u:
                continue
            for v in range(n):
                if v == s or v == u:
                    continue
                for t in range(n):
                    if t == s or t == u or t == v:
                        continue
                    # Check edges exist in potential graph - we just record the path
                    paths.append((s, u, v, t))
    
    # For each path (s,u,v,t), the constraint is: deg[s] != deg[t]
    # This must hold for ALL such paths where the three edges exist.
    
    # Strategy: We'll use a CSP-style approach where we try to build a dense graph.
    # Key insight: if we can find a degree assignment d[0..n-1] and an edge set such that
    # for every path s-u-v-t with all three edges present, d[s] != d[t], we're done.
    
    # Better approach: Greedy construction with smart ordering + local search,
    # but with a much faster validity check using precomputed path structures.
    
    # Precompute for each pair of vertices (a, b) with a < b, which paths have
    # endpoints a and b. Then when checking validity, we just need to verify
    # that for each such path, the three edges exist AND deg[a] != deg[b].
    
    # Actually, the constraint is: for every path s-u-v-t where edges (s,u),(u,v),(v,t)
    # ALL exist, we need deg[s] != deg[t].
    # So a violation occurs when: there exist s,u,v,t all distinct, edges (s,u),(u,v),(v,t)
    # all present, and deg[s] == deg[t].
    
    # Fast check: For a given graph (adjacency sets) and degree array, check all paths.
    # We can speed this up by: for each pair (s,t) with deg[s] == deg[t], check if there
    # exists a path s-u-v-t (s,u,v,t distinct) with all three edges present.
    
    def check_valid_fast(adj, deg):
        """Check validity using degree-group approach."""
        # Group vertices by degree
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)
        
        # For each degree group, check if any two vertices in it are connected
        # by a length-3 path (with distinct intermediate vertices)
        for d, verts in deg_groups.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    # Check if there's a path s-u-v-t with s,u,v,t distinct
                    # and edges (s,u), (u,v), (v,t) all present
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True
    
    # Greedy construction with multiple restarts and local improvement
    def greedy_build(seed_val, initial_edges=None):
        rnd = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
        
        def remove_edge(u, v):
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
            edges.pop()
        
        if initial_edges:
            for u, v in initial_edges:
                add_edge(u, v)
        
        all_e = list(combinations(range(n), 2))
        rnd.shuffle(all_e)
        
        for u, v in all_e:
            if v in adj[u]:
                continue
            add_edge(u, v)
            if check_valid_fast(adj, deg):
                pass  # keep
            else:
                remove_edge(u, v)
        
        # Local improvement: try adding any missing edge
        edge_set = set(edges)
        improved = True
        while improved:
            improved = False
            for u, v in combinations(range(n), 2):
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                add_edge(u, v)
                if check_valid_fast(adj, deg):
                    edge_set.add((u, v))
                    improved = True
                else:
                    remove_edge(u, v)
        
        return edges, adj, deg
    
    # Also try a different strategy: start with a high-degree structure
    # and remove edges to fix violations
    
    def remove_based_build(seed_val):
        rnd = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # Start with complete graph
        for u, v in combinations(range(n), 2):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
        
        edge_set = set(edges)
        
        # Remove edges to fix violations
        improved = True
        while improved:
            improved = False
            # Find a violation
            deg_groups = defaultdict(list)
            for i in range(n):
                deg_groups[deg[i]].append(i)
            
            violation_found = False
            for d, verts in deg_groups.items():
                if len(verts) < 2:
                    continue
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        s, t = verts[i], verts[j]
                        # Find path s-u-v-t
                        for u in adj[s]:
                            if u == t:
                                continue
                            for v in adj[u]:
                                if v == s or v == t:
                                    continue
                                if t in adj[v]:
                                    # Violation found! Remove one of the three edges
                                    # Try removing each and see if it helps
                                    for rem_u, rem_v in [(s, u), (u, v), (v, t)]:
                                        # Check if removing this edge would help
                                        # (i.e., break this particular path)
                                        adj[rem_u].remove(rem_v)
                                        adj[rem_v].remove(rem_u)
                                        deg[rem_u] -= 1
                                        deg[rem_v] -= 1
                                        edges.remove((rem_u, rem_v))
                                        edge_set.discard((rem_u, rem_v))
                                        edge_set.discard((rem_v, rem_u))
                                        improved = True
                                        violation_found = True
                                        break
                                    if violation_found:
                                        break
                            if violation_found:
                                break
                        if violation_found:
                            break
                    if violation_found:
                        break
                if violation_found:
                    break
            
            # If no violation found, we're done
            if not violation_found:
                break
        
        return edges, adj, deg
    
    # Try both approaches
    best_edges = []
    best_count = 0
    
    # Greedy add-based
    for seed_val in range(200):
        edges, adj, deg = greedy_build(seed_val)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
            if best_count >= 70:
                break
    
    # Remove-based
    for seed_val in range(100):
        edges, adj, deg = remove_based_build(seed_val)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
            if best_count >= 70:
                break
    
    # More greedy with different seeds
    if best_count < 70:
        for seed_val in range(200, 500):
            edges, adj, deg = greedy_build(seed_val)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
                if best_count >= 70:
                    break
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
