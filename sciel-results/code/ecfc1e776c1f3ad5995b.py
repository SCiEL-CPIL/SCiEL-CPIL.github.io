import time
import random
import math

def construct():
    n = 15
    deadline = time.time() + 58.0  # Safety margin
    
    # Initialize with K_{7,8}
    # Part A: 0..6, Part B: 7..14
    adj = [0] * n
    deg = [0] * n
    edges = []
    
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges.append([u, v])
            
    def is_valid(adj, deg):
        """
        Checks if the graph defined by adj and deg satisfies the constraint.
        Constraint: For every edge (u,v), the set of degrees of neighbors of u (excluding v)
        and the set of degrees of neighbors of v (excluding u) must be disjoint.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                
                # Get degrees of neighbors of u, excluding v
                set_u = set()
                mask_u = adj[u] & ~(1 << v)
                # Iterate set bits
                m = mask_u
                while m:
                    lsb = m & -m
                    s = lsb.bit_length() - 1
                    set_u.add(deg[s])
                    m ^= lsb
                    
                if not set_u:
                    continue
                
                # Get degrees of neighbors of v, excluding u
                mask_v = adj[v] & ~(1 << u)
                m = mask_v
                while m:
                    lsb = m & -m
                    t = lsb.bit_length() - 1
                    if deg[t] in set_u:
                        return False
                    m ^= lsb
                    
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        # Remove from edges list
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    def get_missing_edges():
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        return missing

    # 1. Greedy Improvement
    # Try to add edges in a specific order or random order
    # We will do a few passes of greedy addition
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        missing = get_missing_edges()
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            add_edge(u, v)
            if is_valid(adj, deg):
                improved = True
            else:
                remove_edge(u, v)
                
    # 2. Simulated Annealing
    # We want to maximize edge count while maintaining validity (violations = 0).
    # If we allow temporary violations, we can escape local optima.
    # Objective: Minimize (violations * 100 + (max_edges - current_edges))
    
    current_edges_count = len(edges)
    best_edges_count = current_edges_count
    best_adj = [x for x in adj]
    best_deg = [x for x in deg]
    best_edges = [e[:] for e in edges]
    
    # Count violations function
    def count_violations(adj, deg):
        cnt = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                
                set_u = set()
                mask_u = adj[u] & ~(1 << v)
                m = mask_u
                while m:
                    lsb = m & -m
                    s = lsb.bit_length() - 1
                    set_u.add(deg[s])
                    m ^= lsb
                    
                if not set_u:
                    continue
                
                mask_v = adj[v] & ~(1 << u)
                m = mask_v
                while m:
                    lsb = m & -m
                    t = lsb.bit_length() - 1
                    if deg[t] in set_u:
                        cnt += 1
                    m ^= lsb
        return cnt

    current_viol = count_violations(adj, deg)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 2 and T > T_min:
        # Pick a random move
        if random.random() < 0.6:
            # Add a random missing edge
            missing = get_missing_edges()
            if not missing:
                break
            u, v = random.choice(missing)
            add_edge(u, v)
            new_viol = count_violations(adj, deg)
            delta_viol = new_viol - current_viol
            delta_edges = 1
            
            # Cost function: we prefer 0 violations. If violations > 0, we penalize heavily.
            # If violations == 0, we prefer more edges.
            # Let's define energy E = violations * 1000 - edges
            # We want to minimize E.
            old_E = current_viol * 1000 - len(edges)
            new_E = new_viol * 1000 - len(edges)
            delta_E = new_E - old_E
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_viol = new_viol
                if new_viol == 0 and len(edges) > best_edges_count:
                    best_edges_count = len(edges)
                    best_adj = [x for x in adj]
                    best_deg = [x for x in deg]
                    best_edges = [e[:] for e in edges]
            else:
                remove_edge(u, v)
        else:
            # Remove a random edge
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            remove_edge(u, v)
            new_viol = count_violations(adj, deg)
            delta_viol = new_viol - current_viol
            delta_edges = -1
            
            old_E = current_viol * 1000 - len(edges)
            new_E = new_viol * 1000 - len(edges)
            delta_E = new_E - old_E
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_viol = new_viol
            else:
                add_edge(u, v)
                
        T *= cooling

    # Final Polish: Greedy add on best solution found
    if time.time() < deadline - 1:
        adj = [x for x in best_adj]
        deg = [x for x in best_deg]
        edges = [e[:] for e in best_edges]
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            missing = get_missing_edges()
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 0.5:
                    break
                add_edge(u, v)
                if is_valid(adj, deg):
                    improved = True
                    if len(edges) > best_edges_count:
                        best_edges_count = len(edges)
                        best_adj = [x for x in adj]
                        best_deg = [x for x in deg]
                        best_edges = [e[:] for e in edges]
                else:
                    remove_edge(u, v)

    # Return the best valid edges found
    # Ensure types are int
    out = [[int(u), int(v)] for u, v in best_edges]
    return out
