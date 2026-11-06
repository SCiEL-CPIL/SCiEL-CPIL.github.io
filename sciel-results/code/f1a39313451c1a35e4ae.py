import time
import random
from itertools import combinations

def is_3_colorable(adj):
    n = len(adj)
    colors = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        
        # Try all 3 colors
        for c in range(3):
            if all(colors[u] != c for u in adj[v]):
                colors[v] = c
                if backtrack(v + 1):
                    return True
                colors[v] = -1
        return False

    # Order vertices by degree descending to prune faster
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Remap indices? No, just use the order in recursion if we change logic.
    # Simpler: Just use standard backtracking with a good vertex order.
    # For N=16, simple backtracking is fast enough.
    
    # Optimization: Use bitmasks for available colors?
    # Let's stick to simple recursion first.
    
    def backtrack_idx(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine forbidden colors
        forbidden = set()
        for u in adj[v]:
            if colors[u] != -1:
                forbidden.add(colors[u])
        
        for c in range(3):
            if c not in forbidden:
                colors[v] = c
                if backtrack_idx(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack_idx(0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 16
    
    # 1. Construct Grötzsch graph on vertices 0..10
    base_adj = [set() for _ in range(n)]
    
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        u = i
        v = (i + 1) % 5
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    # v_i (5+i) connected to u_{i-1} and u_{i+1}
    for i in range(5):
        v = 5 + i
        u_prev = (i - 1) % 5
        u_next = (i + 1) % 5
        base_adj[v].add(u_prev)
        base_adj[u_prev].add(v)
        base_adj[v].add(u_next)
        base_adj[u_next].add(v)
        
    # w (10) connected to all v_i
    for i in range(5):
        v = 5 + i
        base_adj[10].add(v)
        base_adj[v].add(10)
        
    # Vertices 11-15 are initially isolated
    
    all_edges = list(combinations(range(n), 2))
    edge_set = set(all_edges)
    
    best_edge_count = -1
    best_adj = None
    
    # Simulated Annealing / Hill Climbing
    # Start with Grötzsch graph
    curr_adj = [set(s) for s in base_adj]
    curr_edges = set()
    for u in range(n):
        for v in curr_adj[u]:
            if u < v:
                curr_edges.add((u, v))
    
    # Verify initial state is valid (it should be)
    # Grötzsch is triangle-free and 4-chromatic.
    
    temperature = 10.0
    cooling = 0.999
    min_temp = 0.1
    
    # Helper to check validity
    def is_valid(adj, edges):
        # Check triangle-free
        for u, v in edges:
            if adj[u] & adj[v]:
                return False
        # Check 3-colorability
        if is_3_colorable(adj):
            return False
        return True

    # Initial check
    if is_valid(curr_adj, curr_edges):
        best_edge_count = len(curr_edges)
        best_adj = [set(s) for s in curr_adj]
    else:
        # Fallback: Just return Grötzsch
        best_edge_count = len(curr_edges)
        best_adj = [set(s) for s in curr_adj]
        
    while time.time() < deadline:
        if temperature < min_temp:
            break
            
        # Randomly pick an edge to add or remove
        # Bias towards adding edges if we have space
        num_missing = len(edge_set) - len(curr_edges)
        if num_missing > 0 and (random.random() < 0.8 or len(curr_edges) < 20):
            # Try to add an edge
            # Pick random missing edge
            missing_edges = list(edge_set - curr_edges)
            if not missing_edges:
                continue
            u, v = random.choice(missing_edges)
            
            # Check triangle constraint
            if curr_adj[u] & curr_adj[v]:
                continue
            
            # Tentatively add
            curr_adj[u].add(v)
            curr_adj[v].add(u)
            curr_edges.add((u, v))
            
            new_count = len(curr_edges)
            
            # Check 3-colorability
            if is_3_colorable(curr_adj):
                # Invalid, remove
                curr_adj[u].remove(v)
                curr_adj[v].remove(u)
                curr_edges.remove((u, v))
            else:
                # Valid
                if new_count > best_edge_count:
                    best_edge_count = new_count
                    best_adj = [set(s) for s in curr_adj]
        else:
            # Try to remove an edge (to escape local optima)
            if not curr_edges:
                continue
            u, v = random.choice(list(curr_edges))
            
            # Tentatively remove
            curr_adj[u].remove(v)
            curr_adj[v].remove(u)
            curr_edges.remove((u, v))
            
            new_count = len(curr_edges)
            
            # Check if still not 3-colorable
            if is_3_colorable(curr_adj):
                # Became 3-colorable, invalid. Revert.
                curr_adj[u].add(v)
                curr_adj[v].add(u)
                curr_edges.add((u, v))
            else:
                # Still 4-chromatic.
                # Accept with probability exp(-delta / T) if worse
                delta = new_count - best_edge_count # Actually delta is change in objective
                # We want to maximize edges. So if we removed an edge, objective decreased by 1.
                # Accept if random < exp(-1/T)
                if random.random() < (1.0 / (1.0 + temperature)): # Simple acceptance for worse move
                     pass # Keep the removal
                else:
                    # Revert
                    curr_adj[u].add(v)
                    curr_adj[v].add(u)
                    curr_edges.add((u, v))
                    
        temperature *= cooling

    # Final polish: Try to add any missing edges greedily
    for u, v in all_edges:
        if v not in best_adj[u]:
            if not (best_adj[u] & best_adj[v]):
                # Check 3-colorability
                test_adj = [set(s) for s in best_adj]
                test_adj[u].add(v)
                test_adj[v].add(u)
                if not is_3_colorable(test_adj):
                    best_adj[u].add(v)
                    best_adj[v].add(u)
                    best_edge_count += 1

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges
