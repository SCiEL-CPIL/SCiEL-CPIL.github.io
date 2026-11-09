import time
import math
import sys

# Increase recursion limit for B&B
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit for search, 10s buffer
    
    N = 256
    # Squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Valid nodes: 2*v not in squares
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1] # Fallback, though 1+1=2 not square, so 1 is valid
    
    # Map valid nodes to indices 0..M-1
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build adjacency list for valid nodes
    # adj[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        v = valid_nodes[i]
        for j in range(i + 1, M):
            u = valid_nodes[j]
            if v + u in squares:
                adj[i].add(j)
                adj[j].add(i)
        
    # Initial Greedy Solution
    # Sort by degree ascending (low degree first)
    degrees = [len(adj[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x])
    
    current_set = set()
    for i in order:
        if not (adj[i] & current_set):
            current_set.add(i)
            
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Branch and Bound
    # We want to maximize |P|
    # State: P (current set), C (candidates)
    # We use a list for C for efficiency, but sets for intersection checks are easier.
    # For M=200+, sets are fine.
    
    # Heuristic for node selection: Highest degree in induced subgraph?
    # Or simply iterate through C.
    
    # Coloring function for upper bound
    # Greedy coloring of the subgraph induced by C
    def color_bound(candidates):
        # candidates is a set of indices
        # We want to color the graph induced by candidates
        # Color classes are independent sets in the original graph?
        # No, for MIS, the bound comes from coloring the COMPLEMENT?
        # Wait.
        # If we have a set of candidates C, and we want to add nodes to P.
        # The nodes we add must form an independent set in G[C].
        # The size of the maximum independent set in G[C] is at most the number of colors
        # in a coloring of G[C]?
        # No. The size of the maximum independent set is at most the number of colors
        # in a coloring of the COMPLEMENT graph?
        # Let's re-verify.
        # In a graph H, $\alpha(H) \le \chi(H)$.
        # Here H is the graph induced by C on G.
        # So $\alpha(G[C]) \le \chi(G[C])$.
        # So if we color G[C] with k colors, we can add at most k nodes.
        # Yes.
        
        # Greedy coloring of G[C]
        # Order nodes by degree in G[C] descending?
        # For coloring, DSATUR or simple greedy.
        # Simple greedy:
        colors = {}
        # Sort by degree in C descending
        deg_in_c = {v: len(adj[v] & candidates) for v in candidates}
        sorted_c = sorted(candidates, key=lambda x: deg_in_c[x], reverse=True)
        
        for v in sorted_c:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            for u in adj[v]:
                if u in colors:
                    neighbor_colors.add(colors[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
        
        if not colors:
            return 0
        return max(colors.values()) + 1

    # B&B State
    # To avoid copying large sets, we can use a recursive approach with backtracking,
    # but for 256 nodes, copying sets is likely fast enough in Python if depth is not huge.
    # Depth is at most M.
    
    def branch_and_bound(P, C):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if len(P) + len(C) <= best_size:
            return
            
        if not C:
            if len(P) > best_size:
                best_size = len(P)
                best_set = set(P)
            return
            
        # Calculate bound
        # We can use a simpler bound: len(C)
        # But coloring bound is stronger.
        # Let's try coloring bound.
        bound = color_bound(C)
        if len(P) + bound <= best_size:
            return
            
        # Select a node v from C
        # Heuristic: Highest degree in C
        # Calculate degrees in C
        degs = {v: len(adj[v] & C) for v in C}
        v = max(C, key=lambda x: degs[x])
        
        # Branch 1: Include v
        P_new = P | {v}
        C_new = C - (adj[v] | {v})
        branch_and_bound(P_new, C_new)
        
        # Branch 2: Exclude v
        C_new2 = C - {v}
        branch_and_bound(P, C_new2)

    # Initial C is all valid nodes
    C_initial = set(range(M))
    P_initial = set()
    
    # Run B&B
    branch_and_bound(P_initial, C_initial)
    
    # Convert indices back to values
    result = [valid_nodes[i] for i in best_set]
    
    # Final verification
    squares_set = squares
    valid = True
    res_set = set(result)
    for a in result:
        if 2 * a in squares_set:
            valid = False
            break
        for b in result:
            if a <= b and (a + b) in squares_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to greedy if B&B returned invalid (should not happen)
        result = [valid_nodes[i] for i in current_set]
        
    return result
