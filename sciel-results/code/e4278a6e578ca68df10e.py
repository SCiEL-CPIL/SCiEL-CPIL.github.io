"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n = 16
    # Bitmask adjacency
    # adj[u] is an integer where bit v is 1 if edge (u,v) exists
    adj = [0] * n
    
    # Helper to check edge
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    # Helper to add/remove edge
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Count Q3s that would be formed if edge (u,v) was present
    # Assumes (u,v) is NOT currently in adj, or we just check other edges.
    # Actually, it checks if the *other* 11 edges of the Q3 are present.
    # It does not check (u,v).
    def count_q3s_using_edge(u, v):
        # Neighbors of u (excluding v)
        # We need to pick 2 neighbors a, b of u
        # And 2 neighbors c, d of v
        # Such that (a,c) and (b,d) are edges (or a,d and b,c)
        # And there exists f, e completing the cube.
        
        # Get bitmasks of neighbors
        # Note: adj[u] contains v. We should exclude v?
        # In the Q3, u is connected to v, a, b.
        # So a, b are in adj[u] \ {v}.
        neigh_u_mask = adj[u] & ~(1 << v)
        neigh_v_mask = adj[v] & ~(1 << u)
        
        # Get list of neighbors
        neigh_u = []
        mask = neigh_u_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neigh_u.append(idx)
            mask -= lsb
            
        neigh_v = []
        mask = neigh_v_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neigh_v.append(idx)
            mask -= lsb
            
        count = 0
        # Iterate over pairs in neigh_u
        for i in range(len(neigh_u)):
            a = neigh_u[i]
            for j in range(i + 1, len(neigh_u)):
                b = neigh_u[j]
                
                # Iterate over pairs in neigh_v
                for k in range(len(neigh_v)):
                    c = neigh_v[k]
                    for l in range(k + 1, len(neigh_v)):
                        d = neigh_v[l]
                        
                        # Check connectivity patterns
                        # Pattern 1: (a,c) and (b,d)
                        # Pattern 2: (a,d) and (b,c)
                        if not ((has_edge(a, c) and has_edge(b, d)) or (has_edge(a, d) and has_edge(b, c))):
                            continue
                            
                        # Find f: common neighbor of a and b
                        # f must be distinct from u, v, a, b, c, d
                        common_ab = adj[a] & adj[b]
                        # Exclude u, v, a, b, c, d
                        # f cannot be u (u is connected to a,b, but f is 011, u is 000. Not connected in Q3?
                        # Wait, in Q3:
                        # u=000, a=010, b=001.
                        # f=011.
                        # Is f connected to u? 011-000 dist 2. No.
                        # So f is NOT connected to u.
                        # So f is not in adj[u].
                        # But f IS in adj[a] and adj[b].
                        # We just need to ensure f is distinct from the other 6 vertices.
                        forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        candidates_f = common_ab & ~forbidden
                        
                        if not candidates_f:
                            continue
                            
                        # For each candidate f, find e
                        # e is common neighbor of c, d, f
                        # e must be distinct from u, v, a, b, c, d, f
                        # In Q3:
                        # c=110, d=101, f=011.
                        # e=111.
                        # e is connected to c, d, f.
                        # e is NOT connected to u, v, a, b.
                        
                        for f in [x for x in range(n) if (candidates_f >> x) & 1]:
                            forbidden_e = forbidden | (1 << f)
                            common_cdf = adj[c] & adj[d] & adj[f]
                            candidates_e = common_cdf & ~forbidden_e
                            
                            if candidates_e:
                                # Found at least one e.
                                # Does the number of e's matter?
                                # For a fixed set of 7 vertices, e is unique?
                                # In Q3, yes.
                                # In G, there might be multiple e's.
                                # Each valid e forms a distinct Q3.
                                count += candidates_e.bit_count()
                                
        return count

    # Initialize random graph
    random.seed(42)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                add_edge(u, v)

    # Calculate initial violations
    violations = 0
    for u in range(n):
        for v in range(u + 1, n):
            if has_edge(u, v):
                violations += count_q3s_using_edge(u, v)
                
    # Note: This counts each Q3 12 times (once for each edge).
    # This is fine for relative comparison.

    best_violations = violations
    best_adj = adj[:]
    best_edges_count = sum(has_edge(u, v) for u in range(n) for v in range(u+1, n))

    T = 10.0
    T_min = 1e-4
    alpha = 0.999
    steps = 200000 # Increased steps, each step is fast

    for step in range(steps):
        T *= alpha
        if T < T_min:
            T = T_min

        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u

        if has_edge(u, v):
            # Try removing
            # Calculate delta: number of Q3s broken
            # This is count_q3s_using_edge(u, v) assuming edge exists
            # But count_q3s_using_edge checks OTHER edges.
            # So it returns the number of Q3s that currently use (u,v).
            delta = count_q3s_using_edge(u, v)
            
            # If we remove, violations decrease by delta.
            # New violations = violations - delta.
            # We want to minimize violations.
            # If delta > 0, it's an improvement.
            # If delta == 0, it's neutral (but we lose an edge).
            
            if violations > 0:
                # We want to reduce violations.
                if delta > 0:
                    # Always accept if it reduces violations?
                    # Or use SA?
                    # If we always accept, we might get stuck in a local min with low edges.
                    # But we are in "repair" mode.
                    accept = True
                else:
                    # Removing doesn't help. Reject.
                    accept = False
            else:
                # violations == 0
                # Removing an edge (delta must be 0) reduces edge count.
                # Reject.
                accept = False
            
            if accept:
                remove_edge(u, v)
                violations -= delta
                if violations < best_violations:
                    best_violations = violations
                    best_adj = adj[:]
                    best_edges_count = sum(has_edge(u, v) for u in range(n) for v in range(u+1, n))
                elif violations == best_violations:
                    # Tie: prefer more edges?
                    # Current edge count is best_edges_count - 1.
                    # So it's worse.
                    pass
        else:
            # Try adding
            # Calculate delta: number of Q3s formed
            delta = count_q3s_using_edge(u, v)
            
            if violations == 0:
                # We want to keep 0 violations.
                if delta == 0:
                    # Safe to add. Increases edge count.
                    accept = True
                else:
                    # Creates violations. Reject (or SA).
                    # Use SA to allow exploration.
                    if random.random() < np.exp(-delta / T):
                        accept = True
                    else:
                        accept = False
            else:
                # violations > 0
                # Adding an edge increases violations by delta.
                # This is bad.
                # Accept only with very low prob?
                # Or never?
                # If we never accept, we might not escape local minima where we need to add an edge to enable a better removal?
                # Unlikely.
                # Let's use SA.
                if random.random() < np.exp(-delta / T):
                    accept = True
                else:
                    accept = False
            
            if accept:
                add_edge(u, v)
                violations += delta
                if violations < best_violations:
                    best_violations = violations
                    best_adj = adj[:]
                    best_edges_count = sum(has_edge(u, v) for u in range(n) for v in range(u+1, n))
                elif violations == best_violations:
                    # Tie: prefer more edges.
                    # Current edge count is best_edges_count + 1.
                    # So it's better.
                    best_adj = adj[:]
                    best_edges_count += 1

    # Reconstruct best graph
    # best_adj is the adjacency bitmask
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if has_edge(u, v) and (best_adj[u] >> v) & 1:
                 # Wait, best_adj is a snapshot.
                 # We should use best_adj to extract edges.
                 pass
    
    # Extract edges from best_adj
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

# EVOLVE-BLOCK-END
