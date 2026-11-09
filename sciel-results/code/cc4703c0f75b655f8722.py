import random
import time
import math

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Initialize with a random graph or empty graph
    # Let's start with a random subset to have some edges
    initial_edges = set()
    for e in all_edges:
        if random.random() < 0.5:
            initial_edges.add(e)
            
    adj = [[0]*n for _ in range(n)]
    C = [[0]*n for _ in range(n)]
    
    def rebuild_state(edges_set):
        nonlocal adj, C
        adj = [[0]*n for _ in range(n)]
        C = [[0]*n for _ in range(n)]
        for u, v in edges_set:
            adj[u][v] = 1
            adj[v][u] = 1
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w]:
                    C[v][w] += 1
                    C[w][v] += 1
                if adj[v][w]:
                    C[u][w] += 1
                    C[w][u] += 1
        return adj, C

    adj, C = rebuild_state(initial_edges)
    
    def get_violation_count():
        violations = 0
        for u in range(n):
            for v in range(u+1, n):
                if C[u][v] > 2:
                    violations += C[u][v] - 2
        return violations

    current_violations = get_violation_count()
    current_edges = set(initial_edges)
    best_edges = set()
    best_violations = current_violations
    best_count = len(current_edges) if best_violations == 0 else -1
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Precompute neighbors for speed? 
    # We need to quickly add/remove edges and update C.
    
    def add_edge(u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                C[v][w] += 1
                C[w][v] += 1
            if adj[v][w]:
                C[u][w] += 1
                C[w][u] += 1

    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                C[v][w] -= 1
                C[w][v] -= 1
            if adj[v][w]:
                C[u][w] -= 1
                C[w][u] -= 1

    def calculate_delta_violation(u, v, is_addition):
        """
        Calculate change in total violation count if we add/remove edge (u,v).
        This is approximate if we don't track per-pair violations carefully, 
        but we can just recalculate if needed or track the specific pairs affected.
        
        Actually, it's safer to just try the move, calculate new violations, and revert if rejected.
        Given N=20, recalculating violations is O(N^2) = 400 ops, which is fast.
        """
        pass

    # We will use a simple "try and revert" strategy for SA steps.
    # Step:
    # 1. Pick a random move: Add edge, Remove edge, or Swap.
    # 2. Apply move.
    # 3. Calculate new violations.
    # 4. Accept or Reject.
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        move_type = random.random()
        if move_type < 0.33:
            # Add an edge
            u, v = random.choice(all_edges)
            if (u, v) not in current_edges:
                # Try adding
                add_edge(u, v)
                current_edges.add((u, v))
                new_violations = get_violation_count()
                
                # Acceptance
                if new_violations == 0 and len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                elif new_violations <= current_violations or random.random() < math.exp(-(new_violations - current_violations) / T):
                    current_violations = new_violations
                else:
                    # Reject
                    remove_edge(u, v)
                    current_edges.remove((u, v))
        elif move_type < 0.66:
            # Remove an edge
            if current_edges:
                u, v = random.choice(list(current_edges))
                remove_edge(u, v)
                current_edges.remove((u, v))
                new_violations = get_violation_count()
                
                if new_violations <= current_violations or random.random() < math.exp(-(new_violations - current_violations) / T):
                    current_violations = new_violations
                else:
                    # Reject
                    add_edge(u, v)
                    current_edges.add((u, v))
        else:
            # Swap: Remove one, add one
            if current_edges:
                u_out, v_out = random.choice(list(current_edges))
                u_in, v_in = random.choice(all_edges)
                
                # Ensure we don't swap with same edge
                if (u_out, v_out) != (u_in, v_in) and (u_in, v_in) not in current_edges:
                    # Remove out
                    remove_edge(u_out, v_out)
                    current_edges.remove((u_out, v_out))
                    # Add in
                    add_edge(u_in, v_in)
                    current_edges.add((u_in, v_in))
                    
                    new_violations = get_violation_count()
                    
                    if new_violations == 0 and len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
                    
                    if new_violations <= current_violations or random.random() < math.exp(-(new_violations - current_violations) / T):
                        current_violations = new_violations
                    else:
                        # Reject
                        remove_edge(u_in, v_in)
                        current_edges.remove((u_in, v_in))
                        add_edge(u_out, v_out)
                        current_edges.add((u_out, v_out))
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Return best valid edges found
    if best_violations == 0:
        return [list(e) for e in best_edges]
    
    # If no valid graph found (unlikely), return empty or best effort
    # But we must return a valid graph. 
    # We can try to clean up the current state if it's valid
    if current_violations == 0:
        return [list(e) for e in current_edges]
        
    # Fallback: Greedy construction to ensure validity
    adj_f = [[0]*n for _ in range(n)]
    C_f = [[0]*n for _ in range(n)]
    edges_f = []
    random.shuffle(all_edges)
    for u, v in all_edges:
        valid = True
        if C_f[u][v] > 2:
            valid = False
        else:
            for w in range(n):
                if w == u or w == v: continue
                if C_f[u][w] + adj_f[v][w] > 2:
                    valid = False
                    break
                if C_f[v][w] + adj_f[u][w] > 2:
                    valid = False
                    break
        if valid:
            edges_f.append([u, v])
            adj_f[u][v] = 1
            adj_f[v][u] = 1
            for w in range(n):
                if w == u or w == v: continue
                if adj_f[u][w]:
                    C_f[v][w] += 1
                    C_f[w][v] += 1
                if adj_f[v][w]:
                    C_f[u][w] += 1
                    C_f[w][u] += 1
    return edges_f
