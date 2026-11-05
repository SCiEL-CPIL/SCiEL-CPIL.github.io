import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n = 31
    q = 5
    mod = q
    
    # Enumerate all points in PG(2, q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    m = len(edges)
    
    def count_c4s_involving_edge(u, v, adj):
        """Count C4s that include edge (u,v). This is the number of pairs (a,b) 
        where a is neighbor of u (not v), b is neighbor of v (not u), a != b, and b is neighbor of a."""
        count = 0
        nu = adj[u]
        nv = adj[v]
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == a or b == u:
                    continue
                if b in adj[a]:
                    count += 1
        return count

    def count_total_c4(adj):
        """Count all C4s in the graph. Each C4 is counted 4 times (once per edge), so divide by 4."""
        total = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    total += count_c4s_involving_edge(u, v, adj)
        return total // 4

    # Ensure we have exactly 91 edges
    if m < 91:
        # Greedily add edges with minimum C4 contribution
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        while len(edges) < 91:
            best_cost = float('inf')
            best_edge = None
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        cost = count_c4s_involving_edge(i, j, adj)
                        if cost < best_cost:
                            best_cost = cost
                            best_edge = (i, j)
            
            if best_edge:
                u, v = best_edge
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
            else:
                break

    elif m > 91:
        # Greedily remove edges with highest C4 participation
        while len(edges) > 91:
            max_part = -1
            max_edge = None
            for e in edges:
                u, v = e[0], e[1]
                part = count_c4s_involving_edge(u, v, adj)
                if part > max_part:
                    max_part = part
                    max_edge = e
            
            if max_edge:
                u, v = max_edge[0], max_edge[1]
                edges.remove(max_edge)
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                break

    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    # Local Search with incremental C4 counting
    current_adj = [set(s) for s in adj]
    current_score = count_total_c4(current_adj)
    
    best_adj = [set(s) for s in current_adj]
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline - 2.0:
        # Generate random move: remove edge (u,v), add edge (x,y)
        u = random.randint(0, n-1)
        if not current_adj[u]:
            continue
        v = random.choice(list(current_adj[u]))
        
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y:
            continue
        if y in current_adj[x]:
            continue
        
        # Check if this is a valid swap (not removing the same edge we're adding)
        if (u == x and v == y) or (u == y and v == x):
            continue
        
        # Calculate delta C4 count for removing (u,v) and adding (x,y)
        # Delta = -C4s_removed_by_removing_uv + C4s_added_by_adding_xy
        # But we need to be careful: the C4s involving both edges interact
        
        # First, compute C4s involving (u,v) before the move
        c4_uv_before = count_c4s_involving_edge(u, v, current_adj)
        
        # Remove (u,v)
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        
        # Compute C4s involving (x,y) after removal of (u,v)
        c4_xy_after = count_c4s_involving_edge(x, y, current_adj)
        
        # Add (x,y)
        current_adj[x].add(y)
        current_adj[y].add(x)
        
        # The new score change is approximately: -c4_uv_before + c4_xy_after
        # But this is not exact because some C4s may involve both edges
        # For accuracy, let's just compute the full score (it's fast enough for 31 vertices)
        new_score = count_total_c4(current_adj)
        
        # Acceptance
        if new_score < best_score:
            best_score = new_score
            best_adj = [set(s) for s in current_adj]
            
        if new_score <= current_score or random.random() < np.exp((current_score - new_score) / T):
            current_score = new_score
        else:
            # Reject move, revert
            current_adj[x].remove(y)
            current_adj[y].remove(x)
            current_adj[u].add(v)
            current_adj[v].add(u)
            # current_score stays the same
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Convert best_adj back to edges list
    best_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    return best_edges
