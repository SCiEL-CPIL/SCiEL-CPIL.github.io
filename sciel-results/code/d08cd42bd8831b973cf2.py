"""Find dense triangle-free non-3-colorable graph on 16 vertices via local search."""
import time
import random
from itertools import combinations

def _is_triangle_free(adj_sets):
    """Check if graph with adjacency sets is triangle-free."""
    n = len(adj_sets)
    for u in range(n):
        for v in adj_sets[u]:
            if v > u:
                for w in adj_sets[v]:
                    if w > v and w in adj_sets[u]:
                        return False
    return True

def _is_3_colorable(adj_sets):
    """Check if graph is 3-colorable via backtracking with ordering."""
    n = len(adj_sets)
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda x: -len(adj_sets[x]))
    # Map to position in ordering
    pos = {v: i for i, v in enumerate(order)}
    
    color = [-1] * n
    # Forward checking
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        # Determine available colors
        used = set()
        for nb in adj_sets[v]:
            c = color[nb]
            if c != -1:
                used.add(c)
        # Try colors
        for c in range(3):
            if c not in used:
                # Check if this assignment is consistent with already-colored neighbors
                ok = True
                for nb in adj_sets[v]:
                    if pos[nb] < i and color[nb] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack(i + 1):
                        return True
                    color[v] = -1
        return False
    
    return backtrack(0)

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return u * 16 + v

def _key_to_uv(key):
    u = key // 16
    v = key % 16
    return u, v

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with Grötzsch graph (Mycielski of C5) on vertices 0-10, plus isolated 11-15
    # C5: 0-1-2-3-4-0
    # Mycielski: add 5-9 (copies of 0-4), 10 (universal to copies)
    # Edges of C5: (0,1),(1,2),(2,3),(3,4),(4,0)
    # For each edge (u,w) in C5, add (u, 5+w) and (w, 5+u)
    # For each vertex u in C5, add (5+u, 10)
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
    ]
    c5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, w in c5:
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    for u in range(5):
        groetzsch_edges.append([5 + u, 10])
    
    best_edges = set(tuple(sorted(e)) for e in groetzsch_edges)
    best_count = len(best_edges)
    
    # Build adjacency sets for best
    best_adj = [set() for _ in range(16)]
    for u, v in best_edges:
        best_adj[u].add(v)
        best_adj[v].add(u)
    
    # All possible edges
    all_possible = [(u, v) for u in range(16) for v in range(u + 1, 16)]
    
    # Precompute: for each edge (u,v), which other edges would form a triangle
    # If we have edges (u,v), (v,w), (w,u), that's a triangle
    edge_to_triangles = {}
    for (u, v) in all_possible:
        tri = set()
        for w in range(16):
            if w != u and w != v:
                # Triangle u-v-w requires edges (u,v), (v,w), (w,u)
                if (v, w) in set(all_possible) or (w, v) in set(all_possible):
                    # Edge (v,w) or (w,v)
                    e1 = tuple(sorted((v, w)))
                    e2 = tuple(sorted((u, w)))
                    if e1 in set(all_possible) and e2 in set(all_possible):
                        tri.add(e1)
                        tri.add(e2)
        edge_to_triangles[(u, v)] = tri
    
    # Local search
    random.seed(42)
    
    def try_add(adj_sets, edge_set, edge):
        u, v = edge
        if v in adj_sets[u]:
            return False, edge_set
        # Check triangle-free: no common neighbor
        common = adj_sets[u] & adj_sets[v]
        if common:
            return False, edge_set
        # Add edge and check 3-colorability
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        if _is_3_colorable(adj_sets):
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
            return False, edge_set
        else:
            edge_set.add(edge)
            return True, edge_set
    
    def try_remove(adj_sets, edge_set, edge):
        u, v = edge
        if v not in adj_sets[u]:
            return False, edge_set
        adj_sets[u].remove(v)
        adj_sets[v].remove(u)
        edge_set.discard(edge)
        return True, edge_set
    
    # Iterative local search: try to add edges, then try to swap (remove one, add two)
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        # Try to add random edges
        for _ in range(200):
            if time.time() > deadline:
                break
            edge = random.choice(all_possible)
            if edge in best_edges:
                continue
            adj_copy = [s.copy() for s in best_adj]
            set_copy = set(best_edges)
            success, new_set = try_add(adj_copy, set_copy, edge)
            if success:
                best_edges = new_set
                best_adj = adj_copy
                best_count += 1
                improved = True
                break
        
        if not improved:
            # Try swap: remove one edge, add two
            swap_improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Pick an edge to remove
                remove_edge = random.choice(list(best_edges))
                adj_copy = [s.copy() for s in best_adj]
                set_copy = set(best_edges)
                try_remove(adj_copy, set_copy, remove_edge)
                
                # Try to add two edges
                added = 0
                candidates = [e for e in all_possible if e not in set_copy and e != remove_edge]
                random.shuffle(candidates)
                for cand in candidates:
                    if added == 2:
                        break
                    s2, ns2 = try_add(adj_copy, set_copy, cand)
                    if s2:
                        set_copy = ns2
                        added += 1
                
                if added == 2:
                    # We removed 1, added 2, net +1
                    best_edges = set_copy
                    best_adj = adj_copy
                    best_count += 1
                    improved = True
                    swap_improved = True
                    break
        
        if not improved:
            # Try single swap: remove one, add one (no net gain, but may enable further adds)
            swap_improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                remove_edge = random.choice(list(best_edges))
                adj_copy = [s.copy() for s in best_adj]
                set_copy = set(best_edges)
                try_remove(adj_copy, set_copy, remove_edge)
                
                candidates = [e for e in all_possible if e not in set_copy and e != remove_edge]
                random.shuffle(candidates)
                for cand in candidates:
                    s2, ns2 = try_add(adj_copy, set_copy, cand)
                    if s2:
                        set_copy = ns2
                        # Now try to add one more
                        candidates2 = [e for e in all_possible if e not in set_copy]
                        random.shuffle(candidates2)
                        for cand2 in candidates2:
                            s3, ns3 = try_add(adj_copy, set_copy, cand2)
                            if s3:
                                best_edges = ns3
                                best_adj = adj_copy
                                best_count += 1
                                improved = True
                                swap_improved = True
                                break
                        break
                if swap_improved:
                    break
    
    # Final: try systematic addition of all possible edges
    for edge in all_possible:
        if time.time() > deadline:
            break
        if edge in best_edges:
            continue
        adj_copy = [s.copy() for s in best_adj]
        set_copy = set(best_edges)
        success, new_set = try_add(adj_copy, set_copy, edge)
        if success:
            best_edges = new_set
            best_adj = adj_copy
            best_count += 1
    
    return [list(e) for e in best_edges]
