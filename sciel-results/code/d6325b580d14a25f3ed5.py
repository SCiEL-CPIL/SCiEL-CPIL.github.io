"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_3_colorable(adj, n, deadline):
    """Check if graph is 3-colorable using backtracking with deadline."""
    color = [-1] * n
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: sum(adj[v]), reverse=True)
    
    # Precompute adjacency lists for faster neighbor lookup
    adj_list = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                adj_list[u].add(v)
                adj_list[v].add(u)
    
    def backtrack(idx):
        if time.time() > deadline:
            return None  # timeout signal
        if idx == n:
            return True
        v = order[idx]
        # Determine available colors
        used = set()
        for u in adj_list[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(3):
            if c not in used:
                color[v] = c
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    
    # Start with Grötzsch graph on vertices 0-10
    adj = [[0]*n for _ in range(n)]
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        adj[u][v] = adj[v][u] = 1
    for u, w in cycle:
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    
    best_edges = sum(sum(row) for row in adj) // 2
    best_adj = [row[:] for row in adj]
    
    # Candidate edges: all possible edges involving vertices 11-15
    candidate_edges = []
    for u in range(11, 16):
        for v in range(u):
            candidate_edges.append((u, v))
    
    # Also consider removing some edges from the base graph to allow more additions
    # But first, try the greedy approach
    
    # Sort candidates by heuristic: prefer edges between high-degree vertices
    candidate_edges.sort(key=lambda e: (sum(adj[e[0]]) + sum(adj[e[1]]), e[0], e[1]), reverse=True)
    
    # Greedy: try to add each edge if it maintains triangle-freeness and 4-chromaticity
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if adj[u][v]:
            continue
        # Check triangle-freeness: no common neighbor
        common = False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common = True
                break
        if common:
            continue
        # Tentatively add edge
        adj[u][v] = adj[v][u] = 1
        # Check 4-chromaticity (not 3-colorable)
        result = _is_3_colorable(adj, n, deadline)
        if result is None:
            # Timeout during check, revert and stop
            adj[u][v] = adj[v][u] = 0
            break
        if not result:
            best_edges += 1
            best_adj = [row[:] for row in adj]
        else:
            # Revert
            adj[u][v] = adj[v][u] = 0
    
    # Now try a local search phase: try removing one edge and adding up to two
    if time.time() < deadline - 5:
        # Get current edge list
        current_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    current_edges.add((u, v))
        
        # Try all single-edge removals and then greedy re-addition
        for u in range(11, 16):
            for v in range(u):
                if time.time() > deadline:
                    break
                if (u, v) not in current_edges:
                    continue
                
                # Remove edge (u, v)
                adj[u][v] = adj[v][u] = 0
                
                # Try to add edges greedily
                # All candidate edges not in current graph
                new_candidates = []
                for a in range(11, 16):
                    for b in range(a):
                        if (a, b) == (u, v):
                            continue
                        if adj[a][b]:
                            continue
                        # Check triangle-freeness
                        common = False
                        for w in range(n):
                            if w != a and w != b and adj[a][w] and adj[b][w]:
                                common = True
                                break
                        if not common:
                            new_candidates.append((a, b))
                
                # Sort by degree heuristic
                new_candidates.sort(key=lambda e: (sum(adj[e[0]]) + sum(adj[e[1]]), e[0], e[1]), reverse=True)
                
                added = 0
                for a, b in new_candidates:
                    if time.time() > deadline:
                        break
                    # Check triangle-freeness again (graph changed)
                    common = False
                    for w in range(n):
                        if w != a and w != b and adj[a][w] and adj[b][w]:
                            common = True
                            break
                    if common:
                        continue
                    adj[a][b] = adj[b][a] = 1
                    result = _is_3_colorable(adj, n, deadline)
                    if result is None:
                        adj[a][b] = adj[b][a] = 0
                        break
                    if not result:
                        added += 1
                    else:
                        adj[a][b] = adj[b][a] = 0
                
                # If we added more than 1 edge, we improved
                if added > 1:
                    current_edges = set()
                    for a in range(n):
                        for b in range(a+1, n):
                            if adj[a][b]:
                                current_edges.add((a, b))
                    best_edges = len(current_edges)
                    best_adj = [row[:] for row in adj]
                else:
                    # Revert to state before removal
                    adj[u][v] = adj[v][u] = 1
                    for a, b in new_candidates:
                        if adj[a][b] and (a, b) != (u, v):
                            # This was added, need to check if it was part of improvement
                            pass
    
    # Use the best found
    adj = best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
