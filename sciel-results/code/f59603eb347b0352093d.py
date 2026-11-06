import time
import random
import numpy as np
from itertools import combinations

def _has_c4(adj, n):
    """Check if graph has a C4. adj is a list of sets."""
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
    return False

def _find_c4(adj, n):
    """Find one C4 in the graph. Returns (u, a, b, v) such that u-a-b-v-u is a C4, or None."""
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                # Pick two common neighbors a, b
                # We need to ensure a != b and the cycle is simple
                # The cycle is u-a-v-b-u? No, u and v share a and b.
                # So edges (u,a), (a,v), (v,b), (b,u) exist.
                # This forms a 4-cycle u-a-v-b-u.
                # We need to make sure a and b are distinct (they are, from set)
                # and a != u, v (neighbors are distinct from self in simple graph, but adj[u] contains v? 
                # If u~v, then v in adj[u]. If a=v, then u-v-v-b... not simple.
                # adj[u] & adj[v] contains common neighbors. 
                # If u and v are adjacent, v is in adj[u] and u is in adj[v]. 
                # But v is not in adj[v] (no self loops). So v is not in common.
                # So a, b are distinct from u and v.
                # Are a and b distinct? Yes, they are elements of a set.
                # So u-a-v-b-u is a valid C4.
                common_list = list(common)
                a = common_list[0]
                b = common_list[1]
                return (u, a, v, b)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_n = 39
    best_edges = []
    
    # Target: n=47, min_deg >= 8
    # n=46, min_deg >= 7
    
    def try_construct_random(n, target_deg, seed):
        rng = random.Random(seed)
        
        # Try a few different densities
        # Expected degree d = n * p. We want d >= target_deg.
        # Let's try p such that d is between target_deg and target_deg + 2
        for p_offset in [0.0, 0.05, 0.1, 0.2]:
            p = (target_deg + p_offset) / (n - 1)
            if p > 0.5: p = 0.5
            if p < 0.1: p = 0.1
            
            # Initialize random graph
            adj = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if rng.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Check if we already meet the criteria
            def is_valid():
                md = min(len(adj[i]) for i in range(n))
                if md < target_deg:
                    return False
                # Check C4
                if _has_c4(adj, n):
                    return False
                return True
            
            if is_valid():
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
            
            # Local search: Remove C4s, repair degrees
            # We will iterate:
            # 1. Find a C4, remove an edge.
            # 2. If min degree < target, add edges to lowest degree vertices if no C4 created.
            
            max_iter = 500
            for _ in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Step 1: Remove C4s
                c4 = _find_c4(adj, n)
                if c4:
                    u, a, v, b = c4
                    # Edges in C4: (u,a), (a,v), (v,b), (b,u)
                    # Remove one. Prefer removing edge incident to higher degree vertex?
                    # Or just random? Let's try removing edge (u,a) or (a,v) etc.
                    # To minimize degree damage, remove edge between vertices with highest degrees?
                    # Actually, we want to keep degrees high. 
                    # If we remove (u,a), deg(u)-1, deg(a)-1.
                    # Let's pick the edge (x,y) in the C4 such that min(deg(x), deg(y)) is maximized?
                    # No, we want to avoid dropping below target.
                    # Let's just pick a random edge from the C4 to break it.
                    edges_in_c4 = [(u, a), (a, v), (v, b), (b, u)]
                    # Filter edges that exist (they should)
                    rng.shuffle(edges_in_c4)
                    removed = False
                    for x, y in edges_in_c4:
                        if y in adj[x]:
                            # Check if removing this edge keeps degrees >= target_deg - 1?
                            # We can't go below target_deg.
                            # If deg(x) == target_deg or deg(y) == target_deg, removing it drops to target_deg-1.
                            # This is bad. 
                            # So only remove if deg(x) > target_deg and deg(y) > target_deg.
                            if len(adj[x]) > target_deg and len(adj[y]) > target_deg:
                                adj[x].remove(y)
                                adj[y].remove(x)
                                removed = True
                                break
                    if not removed:
                        # We have a C4 but all edges in it are incident to vertices with degree == target_deg.
                        # We are stuck in a "tight" C4.
                        # We might need to add edges elsewhere to increase degrees, then remove C4.
                        # Or accept that this specific random seed/density might fail.
                        # Let's try to add an edge to one of these vertices to increase degree, 
                        # then we can remove a C4 edge safely.
                        # But adding edge might create new C4s.
                        # For now, break and try next seed/density.
                        break
                else:
                    # No C4s. Check min degree.
                    md = min(len(adj[i]) for i in range(n))
                    if md >= target_deg:
                        # Success!
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append([u, v])
                        return edges
                    else:
                        # Need to increase min degree.
                        # Find vertex with min degree.
                        min_deg_vertex = min(range(n), key=lambda i: len(adj[i]))
                        if len(adj[min_deg_vertex]) >= target_deg:
                            # Should not happen if md < target_deg
                            continue
                        
                        # Try to add an edge to min_deg_vertex
                        # Candidates: vertices not in adj[min_deg_vertex]
                        candidates = [v for v in range(n) if v != min_deg_vertex and v not in adj[min_deg_vertex]]
                        rng.shuffle(candidates)
                        
                        added = False
                        for v in candidates:
                            # Check if adding (min_deg_vertex, v) creates a C4
                            # It creates a C4 if there exist a, b such that 
                            # min_deg_vertex ~ a ~ v and a ~ b ~ min_deg_vertex? No.
                            # Adding (u,v) creates C4 if there is a path u-a-b-v of length 3.
                            # i.e., a in N(u), b in N(v), a~b, a!=v, b!=u.
                            u = min_deg_vertex
                            nu = adj[u]
                            nv = adj[v]
                            creates_c4 = False
                            # Optimization: if |nu| or |nv| is small, check all pairs
                            # If large, we can check common neighbors of neighbors?
                            # Simple check:
                            for a in nu:
                                if a == v: continue
                                for b in nv:
                                    if b == u: continue
                                    if a == b: continue
                                    if b in adj[a]:
                                        creates_c4 = True
                                        break
                                if creates_c4: break
                            
                            if not creates_c4:
                                adj[u].add(v)
                                adj[v].add(u)
                                added = True
                                break
                        
                        if not added:
                            # Couldn't add any edge to min_deg_vertex without creating C4
                            # Try next vertex with low degree? Or just fail this attempt.
                            # Let's try to add edge to any vertex with degree < target_deg
                            low_deg_vertices = [i for i in range(n) if len(adj[i]) < target_deg]
                            rng.shuffle(low_deg_vertices)
                            success = False
                            for u in low_deg_vertices:
                                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                                rng.shuffle(candidates)
                                for v in candidates:
                                    nu = adj[u]
                                    nv = adj[v]
                                    creates_c4 = False
                                    for a in nu:
                                        if a == v: continue
                                        for b in nv:
                                            if b == u: continue
                                            if a == b: continue
                                            if b in adj[a]:
                                                creates_c4 = True
                                                break
                                        if creates_c4: break
                                    if not creates_c4:
                                        adj[u].add(v)
                                        adj[v].add(u)
                                        success = True
                                        break
                                if success: break
                            if not success:
                                break # Stuck
            
        return None

    # Try different n values from high to low
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        target_deg = n - 39
        if target_deg < 1:
            target_deg = 1
        
        # Try multiple random seeds
        for seed in range(1000):
            if time.time() > deadline:
                break
            
            edges = try_construct_random(n, target_deg, seed)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                # Found one for this n, try higher n
                break
        
        if time.time() > deadline:
            break
    
    # If we didn't find anything better, return the trivial solution
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
