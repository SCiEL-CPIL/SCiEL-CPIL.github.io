import time
import random

def construct():
    start_time = time.time()
    deadline = 100.0 # Strictly under 120s limit
    
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # common[u][v] stores the number of common neighbors for pair (u, v)
    # We only need to track this for pairs that are NOT edges? 
    # Actually, the constraint is on common neighbors.
    # If u and v are adjacent, they can have common neighbors.
    # The constraint "no K2,3" means no two vertices have 3 common neighbors.
    # So we track common[u][v] for all u < v.
    
    common = [[0] * n for _ in range(n)]
    
    # Initialize with a random sparse graph to avoid symmetry traps
    # Or start empty and build up.
    # Let's start with a random graph with low density to ensure validity.
    # Then hill climb to add edges.
    
    # Helper to add edge and update common counts
    def add_edge(u, v):
        nonlocal adj, common
        if u == v:
            return
        if adj[u] & (1 << v):
            return
        
        # Check if adding this edge violates constraints
        # Adding (u, v) does not change common[u][v].
        # It increases common[v][w] for all w in N(u) (w != v) if u in N(w).
        # It increases common[u][x] for all x in N(v) (x != u) if v in N(x).
        
        # Wait, the constraint is on the FINAL graph.
        # We need to check if the NEW graph has any pair with >= 3 common neighbors.
        # Only pairs involving u or v can change their common neighbor count.
        
        # Pairs (u, w) for w in N(v): common[u][w] might increase if v is in N(w).
        # Pairs (v, w) for w in N(u): common[v][w] might increase if u is in N(w).
        # Pair (u, v): common[u][v] does not change.
        
        # Let's check validity of the move.
        # We need to ensure that for all affected pairs, the new common count < 3.
        
        # Affected pairs:
        # 1. (u, x) where x is a neighbor of v.
        #    If we add (u,v), does common[u][x] change?
        #    common[u][x] = |N(u) ∩ N(x)|.
        #    N(u) gains v. N(x) is unchanged.
        #    So common[u][x] increases by 1 IF v is in N(x).
        #    v is in N(x) iff x is in N(v) (symmetry).
        #    So for all x in N(v), if x != u, common[u][x] increases by 1.
        #    We must ensure common[u][x] + 1 < 3 => common[u][x] <= 1.
        
        # 2. (v, y) where y is a neighbor of u.
        #    Similarly, for all y in N(u), if y != v, common[v][y] increases by 1.
        #    We must ensure common[v][y] + 1 < 3 => common[v][y] <= 1.
        
        # Check conditions
        for x in range(n):
            if x == u: continue
            if adj[v] & (1 << x): # x is in N(v)
                if common[u][x] >= 2:
                    return False
        for y in range(n):
            if y == v: continue
            if adj[u] & (1 << y): # y is in N(u)
                if common[v][y] >= 2:
                    return False
        
        # If valid, apply the change
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts
        # For x in N(v), x != u: common[u][x] += 1
        for x in range(n):
            if x == u: continue
            if adj[v] & (1 << x):
                common[u][x] += 1
                common[x][u] += 1 # Symmetry in common matrix? 
                # common[u][x] is count of common neighbors. It is symmetric.
        
        # For y in N(u), y != v: common[v][y] += 1
        for y in range(n):
            if y == v: continue
            if adj[u] & (1 << y):
                common[v][y] += 1
                common[y][v] += 1
                
        return True

    def remove_edge(u, v):
        nonlocal adj, common
        if not (adj[u] & (1 << v)):
            return
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update common counts (reverse of add)
        for x in range(n):
            if x == u: continue
            if adj[v] & (1 << x):
                common[u][x] -= 1
                common[x][u] -= 1
        
        for y in range(n):
            if y == v: continue
            if adj[u] & (1 << y):
                common[v][y] -= 1
                common[y][v] -= 1

    # Initialize with a random valid graph
    # Try to add random edges until stuck
    random.seed(42)
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        add_edge(u, v)
        
    current_edges = sum(bin(x).count('1') for x in adj) // 2
    best_adj = adj[:]
    best_common = [row[:] for row in common]
    best_edges = current_edges
    
    # Local Search: Try to add edges, if stuck, try 2-swaps
    # We track best valid graph
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() - start_time < deadline:
        improved = False
        
        # Try to add any possible edge
        # To be fast, we can just iterate and try.
        # If we find one, add it and reset loop.
        
        # Shuffle to avoid bias
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                if add_edge(u, v):
                    improved = True
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_common = [row[:] for row in common]
                    break
        
        if improved:
            continue
            
        # If no edge can be added, try a 2-swap (remove 2, add 2)
        # This is more expensive.
        # Pick random edge to remove
        # Then try to add 2 edges
        
        # To keep it simple and fast, we might just restart with a different random seed 
        # or do a limited 2-swap search.
        
        # Let's do a limited 2-swap search
        found_swap = False
        # Sample a few random swaps
        for _ in range(100):
            e1, e2 = random.sample(all_edges, 2)
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if they are existing edges
            if not (adj[u1] & (1 << v1)) or not (adj[u2] & (1 << v2)):
                continue
                
            # Check disjointness
            if len({u1, v1, u2, v2}) < 4:
                continue
                
            # Try swap: remove (u1,v1), (u2,v2), add (u1,u2), (v1,v2)
            # Or (u1,v2), (v1,u2). Let's try one combination.
            new_u1, new_v1 = u1, u2
            new_u2, new_v2 = v1, v2
            
            # Check if new edges exist
            if (adj[new_u1] & (1 << new_v1)) or (adj[new_u2] & (1 << new_v2)):
                continue
                
            # Backup state
            old_adj = adj[:]
            old_common = [row[:] for row in common]
            
            # Remove old edges
            remove_edge(u1, v1)
            remove_edge(u2, v2)
            
            # Add new edges
            ok1 = add_edge(new_u1, new_v1)
            ok2 = add_edge(new_u2, new_v2) if ok1 else False
            
            if ok1 and ok2:
                # Check if we gained edges? No, 2 out 2 in.
                # But maybe it allows more edges later?
                # For now, accept if valid.
                found_swap = True
                # Update best if edge count is same but we are just exploring?
                # Actually, we want to maximize edges. A swap keeps edge count same.
                # It helps escape local optima for ADDING more edges.
                break
            else:
                # Restore state
                adj = old_adj
                common = old_common
                
        if not found_swap:
            # Stuck. Restart with a new random seed?
            # Or just break and return best.
            # Given time, restarting might be better.
            adj = [0] * n
            common = [[0] * n for _ in range(n)]
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                add_edge(u, v)
            current_edges = sum(bin(x).count('1') for x in adj) // 2
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                best_common = [row[:] for row in common]

    # Return best valid graph
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
