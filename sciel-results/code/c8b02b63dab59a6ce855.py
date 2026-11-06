import time
import random

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145
    
    # Construct polarity graph of projective plane of order 5
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0

    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"
    
    min_common = float('inf')
    best_add = None
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                common = len(adj[i] & adj[j])
                if common < min_common:
                    min_common = common
                    best_add = (i, j)
                    
    u_add, v_add = best_add
    edges.add((u_add, v_add))
    adj[u_add].add(v_add)
    adj[v_add].add(u_add)
    
    def binom2(x):
        return x * (x - 1) // 2

    def get_c4_contrib(u, v, adj_sets):
        if u > v: u, v = v, u
        if (u, v) not in edges: # This check is tricky if we pass a modified edge set, but we use adj_sets
            # We need to know if it's an edge in the context of the graph being evaluated
            # For delta calc, we will explicitly handle which edges are present
            pass
        common = len(adj_sets[u] & adj_sets[v])
        return binom2(common)

    # Calculate initial C4
    # Total C4 = 1/4 * sum_{e in E} binom(|N(u) n N(v)|, 2)
    current_c4_sum = 0
    for u, v in edges:
        current_c4_sum += binom2(len(adj[u] & adj[v]))
    current_c4 = current_c4_sum // 4
    
    best_c4 = current_c4
    best_edges = edges.copy()
    best_adj = [set(s) for s in adj]
    best_c4_sum = current_c4_sum

    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute neighbors for fast access? Sets are fine.
    
    iterations = 0
    while time.time() < deadline and iterations < 100000:
        iterations += 1
        
        if not edges:
            break
            
        # Pick random edge to remove
        u1, v1 = random.choice(list(edges))
        
        # Try to find a better swap
        improved = False
        # Limit attempts per iteration to keep pace
        for _ in range(20):
            u2, v2 = random.choice(all_pairs)
            if (u2, v2) in edges or (v2, u2) in edges:
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            # Calculate Delta C4
            # Nodes involved: u1, v1, u2, v2
            involved = {u1, v1, u2, v2}
            
            # Edges affected are those with at least one endpoint in involved
            # We need to sum their contributions in current graph (G)
            sum_old = 0
            for u in involved:
                for v in adj[u]:
                    if u < v:
                        # Check if this edge is affected
                        # It is affected if u in involved or v in involved
                        # Since u is in involved, it is affected.
                        # But we must ensure we don't double count.
                        # We only sum if (u,v) is in edges.
                        if (u, v) in edges:
                            sum_old += binom2(len(adj[u] & adj[v]))
            
            # Now simulate the change to get sum_new
            # Create a local view of adj for involved nodes?
            # Actually, we can just compute the new intersection sizes for affected edges.
            # Affected edges in G' are:
            # 1. All edges in G that were affected (same set) EXCEPT (u1,v1)
            # 2. The new edge (u2,v2)
            
            # Let's just recompute the contribution for all affected edges in the NEW graph
            # New graph edges: (edges - {(u1,v1)}) U {(u2,v2)}
            # New adj: adj[u1] - {v1}, adj[v1] - {u1}, adj[u2] + {v2}, adj[v2] + {u2}
            
            # To avoid modifying global adj, we can create a helper function that takes a "delta" 
            # or just compute the intersection size manually for the few affected edges.
            
            # Affected edges list:
            affected_edges = set()
            for u in involved:
                for v in adj[u]:
                    if u < v:
                        if (u, v) in edges:
                            affected_edges.add((u, v))
            
            # Remove (u1,v1) from affected if it was there (it is, since we picked it from edges)
            affected_edges.discard((u1, v1))
            # Add (u2,v2)
            affected_edges.add((u2, v2))
            
            sum_new = 0
            for u, v in affected_edges:
                # Calculate |N_G'(u) n N_G'(v)|
                # N_G'(x) = N_G(x) 
                #   - {y if (x,y) is removed} 
                #   + {y if (x,y) is added}
                
                # Let's define a function to get N_G'(x)
                def get_n_prime(x):
                    n_x = set(adj[x])
                    # Remove edges incident to x that are removed
                    if x == u1:
                        n_x.discard(v1)
                    elif x == v1:
                        n_x.discard(u1)
                    # Add edges incident to x that are added
                    if x == u2:
                        n_x.add(v2)
                    elif x == v2:
                        n_x.add(u2)
                    return n_x
                
                # We only need N for u and v
                n_u = get_n_prime(u)
                n_v = get_n_prime(v)
                common = len(n_u & n_v)
                sum_new += binom2(common)
            
            new_c4_sum = best_c4_sum - sum_old + sum_new
            new_c4 = new_c4_sum // 4
            
            if new_c4 < best_c4:
                # Accept
                # Update global state
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges.discard((u1, v1))
                edges.add((u2, v2))
                
                best_c4_sum = new_c4_sum
                best_c4 = new_c4
                best_edges = edges.copy()
                best_adj = [set(s) for s in adj]
                improved = True
                break
                
        if time.time() > deadline - 5:
            break

    return list(best_edges)
