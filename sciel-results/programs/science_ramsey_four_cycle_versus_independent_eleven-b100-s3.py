"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n

def _to_bitmask(adj, n):
    """Convert adjacency list of sets to list of bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks

def _max_independent_set_size_bitmask(adj_mask, n, time_limit=1.0):
    """
    Find max independent set size using bitmask branch and bound.
    adj_mask[i] has bit j set if i and j are adjacent.
    Independent set in G is Clique in complement.
    """
    start_time = time.time()
    best = [0]
    
    # Precompute complement masks
    # comp_mask[i] has bit j set if i and j are NOT adjacent (and i != j)
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        # Non-neighbors are (all_mask ^ adj_mask[i] ^ (1 << i))
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    def _search(candidates, current_size, current_vertex):
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound
        # Popcount of candidates is an upper bound on how many more we can add
        # But we can add at most the size of a clique in the induced subgraph.
        # Simple bound: current_size + candidates.bit_count()
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Pick a vertex
        # Heuristic: pick vertex with max degree in the candidate subgraph?
        # For speed, just pick the first one or one with high degree.
        # Let's pick the vertex with the highest degree in the induced subgraph of candidates.
        # This helps prune.
        
        # To save time, we can just pick the first set bit.
        # v = (candidates & -candidates).bit_length() - 1
        
        # Let's try a slightly better heuristic: pick vertex with max number of neighbors in candidates.
        # This is O(n) per step, which is fine for n<40.
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Include v
        # New candidates must be neighbors of v in the complement graph (non-neighbors in G)
        # i.e., they must be in comp_mask[v]
        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1, v)
        
        # Exclude v
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size, v)

    _search((1 << n) - 1, 0, -1)
    return best[0]

def _check_c4_free(adj, n):
    """Check if graph is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _get_compatible_cliques(adj, n, time_limit=5.0):
    """
    Find cliques in the compatibility graph H.
    H has vertices 0..n-1.
    u ~_H v if N_G(u) and N_G(v) are disjoint.
    We want cliques in H, which are sets S of vertices in G that can be neighbors of a new vertex.
    """
    start_time = time.time()
    # Build compatibility graph
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    cliques = []
    
    # Simple Bron-Kerbosch for maximal cliques
    def bronk(R, P, X):
        if time.time() - start_time > time_limit:
            return
        if not P and not X:
            cliques.append(set(R))
            return
        
        # Choose pivot u from P union X to minimize branching
        # u with max degree in P
        u = -1
        max_deg = -1
        P_list = list(P)
        for v in P_list:
            deg = len(P & comp_adj[v])
            if deg > max_deg:
                max_deg = deg
                u = v
        
        if u == -1:
            # P is empty
            if not X:
                 cliques.append(set(R))
            return

        # Branch on vertices in P that are NOT neighbors of u in H
        # Wait, standard Bron-Kerbosch:
        # For each v in P \ N(u):
        #   Bronk(R + v, P & N(v), X & N(v))
        #   P = P - v
        #   X = X + v
        
        P_copy = set(P)
        for v in list(P_copy):
            if v in comp_adj[u]:
                continue
            
            new_R = R | {v}
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            
            bronk(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)

    bronk(set(), set(range(n)), set())
    return cliques

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Build Base Graph
    edges, adj, n = _build_polarity_graph_pg2_5()
    
    # Verify base
    if not _check_c4_free(adj, n):
        return {'n': n, 'edges': edges} # Fallback, shouldn't happen
    
    # Check alpha of base
    adj_mask = _to_bitmask(adj, n)
    alpha = _max_independent_set_size_bitmask(adj_mask, n, time_limit=2.0)
    
    if alpha >= 11:
        return {'n': n, 'edges': edges}

    best_n = n
    best_edges = edges
    best_adj = adj

    # 2. Extension Loop
    while time.time() < deadline - 10:
        # Find compatible cliques in current graph
        cliques = _get_compatible_cliques(adj, n, time_limit=deadline - time.time() - 15)
        
        if not cliques:
            break
            
        # Sort cliques by size descending? Or random?
        # Larger S means smaller U (non-neighbors), potentially smaller alpha(U).
        # But we need alpha(U) <= 9.
        # Let's try larger S first as they "cover" more of the graph.
        cliques.sort(key=lambda s: -len(s))
        
        found_extension = False
        for S in cliques:
            if time.time() > deadline - 10:
                break
                
            # S is the set of neighbors for the new vertex
            # Non-neighbors U = V \ S
            U = set(range(n)) - S
            
            # Check if alpha(G[U]) <= 9
            # We need to build the subgraph induced by U and check its alpha.
            # Map old indices to new 0..|U|-1
            u_list = sorted(list(U))
            u_map = {u: i for i, u in enumerate(u_list)}
            n_u = len(u_list)
            
            # Build adj for subgraph
            sub_adj = [set() for _ in range(n_u)]
            for i in range(n_u):
                for j in range(i+1, n_u):
                    u, v = u_list[i], u_list[j]
                    if v in adj[u]:
                        sub_adj[i].add(j)
                        sub_adj[j].add(i)
            
            sub_adj_mask = _to_bitmask(sub_adj, n_u)
            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=1.0)
            
            if alpha_u <= 9:
                # Valid extension!
                # Add new vertex n
                new_v = n
                new_n = n + 1
                new_adj = [set(s) for s in adj]
                new_adj.append(set(S))
                for u in S:
                    new_adj[u].add(new_v)
                
                new_edges = list(edges)
                for u in S:
                    new_edges.append([u, new_v])
                
                # Update state
                edges = new_edges
                adj = new_adj
                n = new_n
                best_n = n
                best_edges = new_edges
                best_adj = new_adj
                
                found_extension = True
                break # Restart clique search for new graph
        
        if not found_extension:
            # No valid extension found in this round
            # Try randomizing or breaking
            # We can try to shuffle the order of cliques next time?
            # Or just break if we've tried many times.
            # For now, break to save time.
            break

    # Final Validation
    if best_n > 1:
        if not _check_c4_free(best_adj, best_n):
            # Fallback to base if extension broke C4 (should not happen)
            return {'n': n, 'edges': edges}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=5.0)
        
        if alpha_final >= 11:
            # Fallback to base if alpha increased
            return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
