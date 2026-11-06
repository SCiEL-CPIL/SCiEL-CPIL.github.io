import random
import time
import itertools

def _calculate_score(adj, degrees):
    """
    Calculate the maximum degree sum over all triangles in the graph.
    adj: 16x16 numpy array or list of lists (1 if edge, 0 otherwise)
    degrees: list of 16 integers
    """
    n = 16
    max_sum = 0
    # Iterate over all possible triangles
    # i < j < k
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                continue
            for k in range(j + 1, n):
                if adj[i][k] == 1 and adj[j][k] == 1:
                    s = degrees[i] + degrees[j] + degrees[k]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initialize with a random graph or the seed
    # Let's start with the seed to have a valid baseline
    # Seed: K_8,8 + (0,1)
    adj = [[0]*n for _ in range(n)]
    edges = []
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append((u, v))
    adj[0][1] = 1
    adj[1][0] = 1
    edges.append((0, 1))
    
    degrees = [0]*n
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                degrees[u] += 1
                
    current_score = _calculate_score(adj, degrees)
    best_adj = [row[:] for row in adj]
    best_score = current_score
    best_edges = edges[:]
    
    # Simulated Annealing / Random Restart Local Search
    # We will try edge swaps: remove one edge, add one non-edge
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    temperature = 5.0
    cooling_rate = 0.995
    min_temp = 0.1
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        e1_idx = random.randint(0, len(edges)-1)
        e1 = edges[e1_idx]
        
        # Find a random non-edge to add
        # To avoid O(E) check for non-edge, we can pick random pairs and check
        # Or maintain a set of non-edges
        # Given small n, picking random pairs is fine if we limit attempts
        
        added = False
        for _ in range(50): # Try up to 50 times to find a non-edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if adj[u][v] == 0:
                e2 = (u, v)
                # Check if e2 is e1 (should not be since e1 is an edge)
                added = True
                break
        
        if not added:
            continue
            
        # Perform swap
        # Remove e1
        u1, v1 = e1
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        degrees[u1] -= 1
        degrees[v1] -= 1
        edges.pop(e1_idx)
        
        # Add e2
        u2, v2 = e2
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        degrees[u2] += 1
        degrees[v2] += 1
        edges.append(e2)
        
        new_score = _calculate_score(adj, degrees)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_edges = edges[:]
                # If we hit 24, it's likely optimal for this constraint? 
                # Conjecture is ~23.4, so 24 is a strong target.
                # We can keep searching but 24 is excellent.
        else:
            # Revert if not accepted (or accept with probability for SA)
            # Let's use simple hill climbing with occasional random restarts or SA
            delta = new_score - current_score
            if delta > 0:
                # Calculate acceptance probability
                # Since scores are integers, small deltas are common
                # Use simple SA
                if temperature > min_temp:
                    # exp(-delta/T)
                    # delta is at least 1
                    prob = 2.71828 ** (-delta / temperature)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        adj[u2][v2] = 0
                        adj[v2][u2] = 0
                        degrees[u2] -= 1
                        degrees[v2] -= 1
                        edges.pop()
                        
                        adj[u1][v1] = 1
                        adj[v1][u1] = 1
                        degrees[u1] += 1
                        degrees[v1] += 1
                        edges.append(e1)
            else:
                current_score = new_score # Should not happen if delta>0 check covers it, but safe
        
        temperature *= cooling_rate
        if temperature < min_temp:
            # Restart or keep going?
            # Let's just keep going with min_temp to allow some exploration
            temperature = min_temp

    # Convert best_adj to edge list format
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j] == 1:
                final_edges.append([i, j])
                
    return {'n': 16, 'edges': final_edges}
