import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe
    n = 15
    
    # Helper to check if a graph is valid
    def is_valid(adj, deg):
        # For every edge (u, v), check all s in adj[u], t in adj[v]
        # s != v, t != u, s != t
        # Must have deg[s] != deg[t]
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # check each edge once? No, adj is directed in list sense, but graph is undirected.
                # Actually, let's just iterate all u, v in adj[u]
                # This might be O(E * D^2). With n=15, max E=105, D<15. 105*225 ~ 23k ops. Very fast.
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def get_deg(adj):
        return [len(a) for a in adj]

    def try_add_edge(adj, u, v):
        if u == v: return False
        if v in adj[u]: return False
        
        # Tentatively add
        adj[u].append(v)
        adj[v].append(u)
        deg = get_deg(adj)
        
        if is_valid(adj, deg):
            return True
        else:
            # Revert
            adj[u].remove(v)
            adj[v].remove(u)
            return False

    def try_remove_edge(adj, u, v):
        if v not in adj[u]: return False
        adj[u].remove(v)
        adj[v].remove(u)
        return True

    def count_edges(adj):
        return sum(len(a) for a in adj) // 2

    # Initialize with empty graph
    adj = [[] for _ in range(n)]
    best_adj = [list(a) for a in adj]
    best_count = 0
    
    # Random seed for reproducibility if needed, but let's just use default
    rng = random.Random(42)
    
    # Phase 1: Greedy addition
    # Try to add all possible edges in random order
    edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if try_add_edge(adj, u, v):
            pass
        if time.time() > deadline - 10: break
        
    current_count = count_edges(adj)
    if current_count > best_count:
        best_adj = [list(a) for a in adj]
        best_count = current_count

    # Phase 2: Local Search
    # Randomly pick an edge to remove, then try to add random edges
    # Or: Pick a non-edge to add. If fails, pick an edge to remove, then try to add.
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline - 5:
        # Try to add a random non-edge
        success = False
        attempts = 0
        while attempts < 20 and not success and time.time() < deadline - 5:
            u, v = rng.choice(all_possible_edges)
            if v not in adj[u]:
                if try_add_edge(adj, u, v):
                    success = True
                    cur = count_edges(adj)
                    if cur > best_count:
                        best_adj = [list(a) for a in adj]
                        best_count = cur
            attempts += 1
            
        if not success:
            # Try a move: remove an edge, then add two edges? 
            # Simpler: Remove a random edge, then try to add the same one back? No.
            # Remove a random edge, then try to add random non-edges.
            current_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if current_edges:
                rem_u, rem_v = rng.choice(current_edges)
                try_remove_edge(adj, rem_u, rem_v)
                
                # Now try to add some edges
                added = 0
                for _ in range(5):
                    if time.time() > deadline - 5: break
                    u, v = rng.choice(all_possible_edges)
                    if v not in adj[u]:
                        if try_add_edge(adj, u, v):
                            added += 1
                            
                cur = count_edges(adj)
                if cur > best_count:
                    best_adj = [list(a) for a in adj]
                    best_count = cur
                # We keep the modified adj (even if worse) to explore different regions?
                # Usually hill climbing keeps current if better, or reverts if worse.
                # Let's revert if worse to stay on the hill, unless we want simulated annealing.
                # Given time, simple hill climbing on edge count is safer.
                # But we just removed an edge. If we didn't add enough to compensate, we are lower.
                # Let's just accept the state if it's valid (it is) and see if we can climb from there.
                # Actually, standard local search: accept if better. If not, revert?
                # If we revert, we are back to previous state. 
                # Let's just keep the state if it's valid. The "best" is tracked separately.
                # This is a random walk on the space of valid graphs.

    # Extract best edges
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    # Ensure types are correct
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges

# EVOLVE-BLOCK-END
