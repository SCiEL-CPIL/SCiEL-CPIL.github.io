import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    best_edges = None
    best_count = 0
    
    def update_best(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]

    # Strategy A: Greedy Insertion (Current approach)
    random.seed(42)
    
    # Limit time for Strategy A
    time_limit_A = 10.0
    
    max_restarts_A = 500
    for restart in range(max_restarts_A):
        if time.time() - start_time > DEADLINE - time_limit_A:
            break
        
        candidate_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(candidate_edges)
        
        edge_list = []
        neighbors = {v: set() for v in range(n)}
        edge_color = {}
        
        for (u, v) in candidate_edges:
            if time.time() - start_time > DEADLINE:
                break
            
            forbidden = set()
            can_add = True
            
            for w in neighbors[u]:
                if w == v:
                    continue
                for x in neighbors[v]:
                    if x == u or x == w:
                        continue
                    
                    key_wx = frozenset((w, x))
                    if key_wx not in edge_color:
                        continue
                    
                    key_uw = frozenset((u, w))
                    key_xv = frozenset((x, v))
                    
                    c_1 = edge_color[key_uw]
                    c_2 = edge_color[key_wx]
                    c_3 = edge_color[key_xv]
                    
                    if len({c_1, c_2, c_3}) < 3:
                        can_add = False
                        break
                
                if not can_add:
                    break
            
            if not can_add:
                continue
            
            for w in neighbors[u]:
                if w == v:
                    continue
                for x in neighbors[v]:
                    if x == u or x == w:
                        continue
                    
                    key_wx = frozenset((w, x))
                    if key_wx not in edge_color:
                        continue
                    
                    key_uw = frozenset((u, w))
                    key_xv = frozenset((x, v))
                    
                    c_1 = edge_color[key_uw]
                    c_2 = edge_color[key_wx]
                    c_3 = edge_color[key_xv]
                    
                    forbidden.add(c_1)
                    forbidden.add(c_2)
                    forbidden.add(c_3)
            
            available = [c for c in colors if c not in forbidden]
            if not available:
                continue
            
            c = random.choice(available)
            key = frozenset((u, v))
            edge_color[key] = c
            edge_list.append([u, v, c])
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        update_best(edge_list)

    # Strategy B: Bipartite K_10,10 with Greedy Removal
    U = list(range(10))
    V = list(range(10, 20))
    
    time_limit_B = 10.0
    
    for restart in range(1000):
        if time.time() - start_time > DEADLINE - time_limit_B:
            break
        
        # Random coloring strategy
        # c(u, v) = (a * u + b * v + k) % 20
        a = random.randint(1, 19)
        b = random.randint(1, 19)
        k = random.randint(0, 19)
        
        # Ensure a, b are not 0 to avoid degenerate cases, though mod 20 handles it
        # Actually, if a=0, c depends only on v.
        
        edges = []
        edge_colors = {}
        
        for u in U:
            for v in V:
                c = (a * u + b * v + k) % 20
                edges.append([u, v, c])
                edge_colors[(u, v)] = c
                edge_colors[(v, u)] = c
        
        # Find bad cycles
        bad_cycles = []
        for i, u1 in enumerate(U):
            for u2 in U[i+1:]:
                for j, v1 in enumerate(V):
                    for v2 in V[j+1:]:
                        c1 = edge_colors[(u1, v1)]
                        c2 = edge_colors[(u2, v1)]
                        c3 = edge_colors[(u2, v2)]
                        c4 = edge_colors[(u1, v2)]
                        
                        if len({c1, c2, c3, c4}) < 4:
                            bad_cycles.append([(u1, v1), (u2, v1), (u2, v2), (u1, v2)])
        
        if not bad_cycles:
            update_best(edges)
            continue
            
        # Greedy removal
        # Map edge to set of cycle indices it belongs to
        edge_to_cycles = {e: [] for e in edges}
        # Actually, edges are tuples (u,v). In bad_cycles, they are also tuples.
        # Let's use a dict of frozensets for edges to be consistent?
        # No, (u,v) is fine if we are careful.
        
        # Let's use a list of sets for cycles
        # cycle_edges[i] = set of edges in cycle i
        cycle_edges = [set(c) for c in bad_cycles]
        
        # edge_freq: edge -> count of cycles
        edge_freq = {}
        for ce in cycle_edges:
            for e in ce:
                edge_freq[e] = edge_freq.get(e, 0) + 1
        
        active_cycles = [True] * len(cycle_edges)
        num_active = len(cycle_edges)
        
        while num_active > 0:
            # Find edge with max freq among active cycles
            best_edge = None
            max_hits = 0
            for e, cnt in edge_freq.items():
                if cnt > max_hits:
                    max_hits = cnt
                    best_edge = e
            
            if best_edge is None:
                break
                
            # Remove best_edge
            # Update active_cycles
            for i, ce in enumerate(cycle_edges):
                if active_cycles[i] and best_edge in ce:
                    active_cycles[i] = False
                    num_active -= 1
                    # Decrement freq for other edges in this cycle
                    for e in ce:
                        if e != best_edge:
                            edge_freq[e] -= 1
            
            # Remove best_edge from edges list?
            # We just need to know which edges to keep.
            # Let's mark it for removal.
            if best_edge in edge_freq:
                del edge_freq[best_edge]
                
        # Construct final edges
        # We need to know which edges were removed.
        # The loop above removes them from edge_freq.
        # But we started with all edges in edge_freq?
        # No, edge_freq only contains edges in bad cycles.
        # Edges not in bad cycles are safe.
        
        removed_edges = set()
        # We need to track which edges were removed.
        # Let's re-implement the removal to track removed edges.
        
        # Re-do removal with tracking
        edge_freq = {}
        for ce in cycle_edges:
            for e in ce:
                edge_freq[e] = edge_freq.get(e, 0) + 1
                
        active_cycles = [True] * len(cycle_edges)
        num_active = len(cycle_edges)
        removed_edges = set()
        
        while num_active > 0:
            best_edge = None
            max_hits = 0
            for e, cnt in edge_freq.items():
                if cnt > max_hits:
                    max_hits = cnt
                    best_edge = e
            
            if best_edge is None:
                break
                
            removed_edges.add(best_edge)
            
            for i, ce in enumerate(cycle_edges):
                if active_cycles[i] and best_edge in ce:
                    active_cycles[i] = False
                    num_active -= 1
                    for e in ce:
                        if e != best_edge:
                            edge_freq[e] -= 1
                            
            if best_edge in edge_freq:
                del edge_freq[best_edge]
                
        final_edges = []
        for e in edges:
            u, v, c = e
            if (u, v) not in removed_edges and (v, u) not in removed_edges:
                final_edges.append(e)
                
        update_best(final_edges)

    if best_edges is None:
        return [[0, v, 0] for v in range(1, n)]
    
    return best_edges
