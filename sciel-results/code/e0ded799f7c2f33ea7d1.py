"""Construct a dense K3,3-free graph on 20 vertices using bitmask local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 20
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m_total = len(all_edges)  # 190
    
    # Map edge to index
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # Bitmask for adjacency
    # adj[i] is a bitmask of neighbors of i
    
    def is_k33_free(adj, n=20):
        """Check if graph with adjacency bitmasks is K3,3-free.
        K3,3 exists iff there exist 3 vertices with >=3 common neighbors.
        """
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                # Count bits in common
                if common & (common - 1):
                    # Check if popcount >= 3
                    c = common
                    if c & (c - 1):
                        c2 = c & (c - 1)
                        if c2 & (c2 - 1):
                            return False
        return True
    
    def count_edges(adj, n=20):
        """Count edges from adjacency bitmasks."""
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    def find_k33_violation(adj, n=20):
        """Find a K3,3 violation: returns (i, j, k, l, m, p) or None.
        i,j,k are the 3 vertices with common neighbors l,m,p.
        """
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):
                    c2 = common & (common - 1)
                    if c2 & (c2 - 1):
                        # Find 3 bits in common
                        bits = []
                        c = common
                        while c and len(bits) < 3:
                            low = c & (-c)
                            bits.append(low.bit_length() - 1)
                            c ^= low
                        if len(bits) == 3:
                            k = bits[2]
                            # Need a third vertex from {i,j,k} perspective
                            # Actually K3,3 means 3 vertices on one side, 3 on other
                            # Here i,j have common neighbors. We need a third vertex 
                            # that also shares those neighbors, OR we check triples.
                            # Let's check: do i,j,k have 3 common neighbors?
                            # Actually the condition is: no 3 vertices have 3 common neighbors.
                            # So we need to check all triples.
                            pass
        
        # Better approach: check all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if common & (common - 1):
                        c2 = common & (common - 1)
                        if c2 & (c2 - 1):
                            # Found 3+ common neighbors
                            bits = []
                            c = common
                            while c and len(bits) < 3:
                                low = c & (-c)
                                bits.append(low.bit_length() - 1)
                                c ^= low
                            if len(bits) == 3:
                                return (i, j, k, bits[0], bits[1], bits[2])
        return None
    
    def find_violation_fast(adj, n=20):
        """Faster violation check using pair-based approach.
        For each pair (i,j), compute common neighbors. If any pair has >=3 common neighbors,
        we need to check if there's a third vertex sharing those.
        Actually the constraint is: no three vertices have three common neighbours.
        So for each triple (i,j,k), |N(i) ∩ N(j) ∩ N(k)| < 3.
        
        Alternative: for each pair (i,j), let C = N(i)∩N(j). For each c in C,
        if there exist two other vertices i',j' such that c is in N(i')∩N(j')∩N(i)∩N(j)... 
        This is getting complex. Let's just do the triple check but optimize.
        """
        # Precompute pair intersections
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                if not (common_ij & (common_ij - 1)):
                    continue  # fewer than 2 common neighbors
                # For each k > j, check if adj[k] shares >=3 with common_ij
                # Actually we need |N(i)∩N(j)∩N(k)| >= 3
                # = |common_ij ∩ N(k)| >= 3
                for k in range(j+1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk & (common_ijk - 1):
                        c2 = common_ijk & (common_ijk - 1)
                        if c2 & (c2 - 1):
                            # Found violation
                            bits = []
                            c = common_ijk
                            while c and len(bits) < 3:
                                low = c & (-c)
                                bits.append(low.bit_length() - 1)
                                c ^= low
                            if len(bits) == 3:
                                return (i, j, k, bits[0], bits[1], bits[2])
        return None
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Strategy: Start with a dense random graph, then remove edges to eliminate K3,3
    # Use multiple restarts with different random seeds
    
    best_adj = None
    best_edge_count = 0
    
    # Try several random initializations
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random dense graph with edge probability p
        p = 0.6 + 0.3 * (trial % 10) / 10  # vary density
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove edges to fix K3,3 violations
        iterations = 0
        max_iterations = 5000
        while iterations < max_iterations:
            if time.time() > deadline - 10:
                break
            iterations += 1
            viol = find_violation_fast(adj)
            if viol is None:
                break
            i, j, k, l, m, p2 = viol
            # Remove one of the edges among the 6 vertices of K3,3
            # To break the K3,3, remove an edge from the bipartite structure
            # The K3,3 has left={i,j,k}, right={l,m,p2}
            # Remove a random edge between left and right
            candidates = [(i,l),(i,m),(i,p2),(j,l),(j,m),(j,p2),(k,l),(k,m),(k,p2)]
            # Only remove edges that exist
            existing = []
            for (u,v) in candidates:
                if (adj[u] >> v) & 1:
                    existing.append((u,v))
            if existing:
                u, v = random.choice(existing)
                remove_edge(adj, u, v)
        
        ec = count_edges(adj)
        if ec > best_edge_count:
            best_edge_count = ec
            best_adj = adj[:]
        
        # Local search: try to add edges back
        if best_adj is not None:
            adj = best_adj[:]
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try adding each missing edge
                for i in range(n):
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            break
                        if not ((adj[i] >> j) & 1):
                            # Try adding edge (i,j)
                            add_edge(adj, i, j)
                            if find_violation_fast(adj) is None:
                                best_adj = adj[:]
                                best_edge_count += 1
                                improved = True
                            else:
                                remove_edge(adj, i, j)
                    if time.time() > deadline - 5:
                        break
        
        # Edge swap local search
        if best_adj is not None and time.time() < deadline - 5:
            adj = best_adj[:]
            ec = count_edges(adj)
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                # Pick a random existing edge to remove
                existing_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            existing_edges.append((i,j))
                if not existing_edges:
                    break
                u, v = random.choice(existing_edges)
                remove_edge(adj, u, v)
                # Try to add a random missing edge
                missing_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not ((adj[i] >> j) & 1):
                            missing_edges.append((i,j))
                if not missing_edges:
                    add_edge(adj, u, v)
                    break
                u2, v2 = random.choice(missing_edges)
                add_edge(adj, u2, v2)
                if find_violation_fast(adj) is None:
                    ec_new = count_edges(adj)
                    if ec_new >= ec:
                        ec = ec_new
                        best_adj = adj[:]
                        best_edge_count = max(best_edge_count, ec)
                    else:
                        # Revert
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u, v)
                else:
                    # Revert
                    remove_edge(adj, u2, v2)
                    add_edge(adj, u, v)
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
