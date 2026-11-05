"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def _score(graph_edges, n=16):
    """Compute the largest triangle degree sum efficiently."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in graph_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    max_sum = 0
    # Iterate over all pairs (u, v) with u < v
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Find common neighbors w > v to avoid double counting
            # w must be in adj[u] and adj[v]
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _local_search(edges, n=16, time_limit=5.0, seed=None):
    """
    Perform local search by swapping edges to reduce the maximum triangle degree sum.
    Strategy: Identify the triangle with the max sum. Try to break it by removing one of its edges 
    and adding an edge that doesn't create a worse triangle, or simply accept if the score drops.
    """
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    current_edges = [e[:] for e in edges]
    current_score = _score(current_edges, n)
    best_edges = [e[:] for e in current_edges]
    best_score = current_score
    
    # Pre-compute adjacency for speed
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
    # Find the current "worst" triangle to target
    def find_worst_triangle():
        worst = None
        worst_sum = -1
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        s = deg[u] + deg[v] + deg[w]
                        if s > worst_sum:
                            worst_sum = s
                            worst = (u, v, w)
        return worst, worst_sum

    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < T_min:
            break
            
        # Pick a random edge to remove
        if not current_edges:
            break
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Find a candidate edge to add
        # Heuristic: Prefer adding edges between vertices with lower degrees to keep degrees balanced
        # Or, try to "fix" the worst triangle by removing one of its edges
        worst_tri, _ = find_worst_triangle()
        
        # Strategy 1: Remove an edge from the worst triangle
        if worst_tri and random.random() < 0.5:
            # Pick one of the 3 edges in the worst triangle to remove
            tri_edges = [(worst_tri[0], worst_tri[1]), (worst_tri[0], worst_tri[2]), (worst_tri[1], worst_tri[2])]
            rem_u, rem_v = random.choice(tri_edges)
            # Ensure this edge exists
            if (rem_u, rem_v) not in [(a,b) for a,b in current_edges] and (rem_v, rem_u) not in [(a,b) for a,b in current_edges]:
                 # Fallback to random removal if logic fails, but we know it's in the triangle so it should exist
                 pass
            # Find a non-edge to add
            # We want to add an edge that doesn't create a new high-degree triangle
            # Try random non-edges
            for _ in range(10):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a == b: continue
                a, b = min(a, b), max(a, b)
                if (a, b) in adj[a]: continue
                if (a, b) == (rem_u, rem_v) or (a, b) == (rem_v, rem_u): continue
                
                # Tentatively apply swap
                # Remove (rem_u, rem_v)
                adj[rem_u].remove(rem_v)
                adj[rem_v].remove(rem_u)
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                
                # Add (a, b)
                adj[a].add(b)
                adj[b].add(a)
                deg[a] += 1
                deg[b] += 1
                
                new_score = _score([(a,b) for a in range(n) for b in adj[a] if a < b], n)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    # Accept
                    # Update current_edges list
                    # This is inefficient, let's just rebuild from adj for best tracking? 
                    # Actually, maintaining the list is important for the return value.
                    # Let's just record the move.
                    
                    # Update best if better
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = [[min(x,y), max(x,y)] for x in range(n) for y in adj[x] if x < y]
                    
                    # Update current state variables
                    current_score = new_score
                    # Update current_edges list
                    # Remove old, add new
                    current_edges.remove([min(rem_u, rem_v), max(rem_u, rem_v)])
                    current_edges.append([a, b])
                    break
                else:
                    # Reject: Undo
                    adj[a].remove(b)
                    adj[b].remove(a)
                    deg[a] -= 1
                    deg[b] -= 1
                    adj[rem_u].add(rem_v)
                    adj[rem_v].add(rem_u)
                    deg[rem_u] += 1
                    deg[rem_v] += 1
        else:
            # Strategy 2: Random swap
            for _ in range(10):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a == b: continue
                a, b = min(a, b), max(a, b)
                if (a, b) in adj[a]: continue
                
                # Remove (u, v)
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                
                # Add (a, b)
                adj[a].add(b)
                adj[b].add(a)
                deg[a] += 1
                deg[b] += 1
                
                new_score = _score([(x,y) for x in range(n) for y in adj[x] if x < y], n)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = [[min(x,y), max(x,y)] for x in range(n) for y in adj[x] if x < y]
                    current_score = new_score
                    current_edges.remove([min(u, v), max(u, v)])
                    current_edges.append([a, b])
                    break
                else:
                    # Undo
                    adj[a].remove(b)
                    adj[b].remove(a)
                    deg[a] -= 1
                    deg[b] -= 1
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1

    return best_edges, best_score

def _generate_initial_graph(n=16, target_edges=65, seed=None):
    """Generate a random graph with target edges, biased towards Turan structure."""
    if seed is not None:
        random.seed(seed)
        
    # Start with Turan T_2(16) = K_{8,8} (64 edges)
    part_a = list(range(8))
    part_b = list(range(8, 16))
    
    # Shuffle vertices to randomize partition
    all_v = list(range(n))
    random.shuffle(all_v)
    part_a = all_v[:8]
    part_b = all_v[8:]
    
    edges = set()
    for u in part_a:
        for v in part_b:
            edges.add((min(u, v), max(u, v)))
            
    # Need 1 more edge
    # Add a random internal edge
    while True:
        a = random.choice(part_a)
        b = random.choice(part_a)
        if a == b: continue
        e = (min(a, b), max(a, b))
        if e not in edges:
            edges.add(e)
            break
            
    # Now we have 65 edges. 
    # We can also try to randomize by swapping some cross edges for internal ones if we want diversity,
    # but for now, let's just shuffle the edge list.
    edge_list = [list(e) for e in edges]
    random.shuffle(edge_list)
    return edge_list

def construct():
    """
    Construct a graph on 16 vertices with 65 edges minimizing the max triangle degree sum.
    """
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_edges = None
    
    # 1. Baseline: Standard Turan + 1
    edges_seed = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    score_seed = _score(edges_seed, n)
    if score_seed < best_score:
        best_score = score_seed
        best_edges = [e[:] for e in edges_seed]
        
    # 2. Local Search from multiple random Turan-like starts
    # We have about 100s. Let's run several independent local searches.
    # Each local search has a time limit.
    
    start_time = time.time()
    time_limit_per_run = 8.0
    max_runs = 10
    
    for i in range(max_runs):
        if time.time() - start_time > 110.0:
            break
            
        seed = i * 100 + 42
        init_edges = _generate_initial_graph(n, target_edges, seed=seed)
        
        # Run local search
        try:
            opt_edges, opt_score = _local_search(init_edges, n, time_limit=time_limit_per_run, seed=seed)
            if opt_score < best_score:
                best_score = opt_score
                best_edges = [e[:] for e in opt_edges]
        except Exception as e:
            pass
            
    # 3. If we have time left, run another round with different strategy?
    # For now, return the best found.
    
    if best_edges is None:
        best_edges = edges_seed
        
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
