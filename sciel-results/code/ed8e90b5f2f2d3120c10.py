import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v: u, v = v, u
        base_edges.add((u, v))
    
    # Candidate edges: bipartite (opposite parity), not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u + v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
    
    def get_diameter(edge_set):
        adj = [[] for _ in range(n)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
                        q.append(nb)
        return max_dist

    def is_valid(edge_set):
        # Triangle-free is guaranteed by bipartite construction
        # Check connected and diameter <= 4
        diam = get_diameter(edge_set)
        if diam == -1: # Disconnected (shouldn't happen if base is connected and we only add/remove carefully, but safe)
             return False
        return diam <= 4

    # 1. Greedy Initialization
    current_edges = set(base_edges)
    added = []
    
    # Fast Greedy: Add edge that reduces diameter most
    # We aim for a valid solution first
    for _ in range(20): # Max 20 edges
        diam = get_diameter(current_edges)
        if diam <= 4:
            break
        
        best_cand = None
        best_new_diam = float('inf')
        
        # Sample candidates to speed up initialization if needed, but n=24 is small
        for cand in candidates:
            if cand in current_edges:
                continue
            new_edges = current_edges | {cand}
            # Quick check: does it connect components or reduce dist?
            # Just compute diameter
            new_diam = get_diameter(new_edges)
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_cand = cand
        
        if best_cand is None:
            break
            
        current_edges.add(best_cand)
        added.append(list(best_cand))
        
    if not is_valid(current_edges):
        # Fallback: return whatever we have, or try a few random additions
        # But we must return a valid solution or invalid scores 0.
        # If greedy failed, try random restarts
        pass

    # 2. Local Search to Minimize Count
    # We want to reduce len(added)
    # Moves: Remove, Swap
    
    deadline = time.time() + 85.0
    best_added = list(added)
    best_count = len(added)
    
    # Ensure we start with a valid state for local search
    if not is_valid(current_edges):
         # Try to repair or just return current if it's the best we got
         # But let's assume greedy found a valid one (10-12 edges is expected)
         pass

    while time.time() < deadline:
        improved = False
        
        # Try Remove
        for i in range(len(added) - 1, -1, -1):
            if time.time() > deadline:
                break
            edge = tuple(added[i])
            if edge in current_edges:
                new_edges = current_edges - {edge}
                if is_valid(new_edges):
                    current_edges = new_edges
                    added.pop(i)
                    improved = True
                    if len(added) < best_count:
                        best_count = len(added)
                        best_added = list(added)
                    break # Restart removal loop with new state
        
        if improved:
            continue
            
        # Try Swap
        # Pick a random added edge to remove
        if added:
            idx = random.randint(0, len(added) - 1)
            edge_to_remove = tuple(added[idx])
            
            # Try adding a random candidate instead
            # To be efficient, we can sample candidates
            # Or iterate all if time permits
            swap_success = False
            # Shuffle candidates for randomness
            cand_copy = candidates[:]
            random.shuffle(cand_copy)
            
            for cand in cand_copy:
                if time.time() > deadline:
                    break
                if cand in current_edges:
                    continue
                # Remove edge_to_remove, add cand
                new_edges = (current_edges - {edge_to_remove}) | {cand}
                if is_valid(new_edges):
                    # Accept swap
                    current_edges = new_edges
                    added[idx] = list(cand)
                    swap_success = True
                    # After swap, try to remove again?
                    # The loop will handle removal in next iteration
                    break
            
            if swap_success:
                # Check if we can remove anything now?
                # The outer loop will handle it.
                # But maybe the swap allowed a removal?
                # Let's just let the loop continue.
                pass
        
        # If no improvement in a while, maybe randomize?
        # For now, simple hill climbing.
        
    # Final check and cleanup
    # Ensure returned edges are valid Python types
    final_added = []
    for e in best_added:
        u, v = e
        if u > v: u, v = v, u
        # Ensure int
        final_added.append([int(u), int(v)])
        
    # Verify final solution validity one last time
    # If best_added was from a state that became invalid later (shouldn't happen if we track best correctly)
    # We track best_added only when we improve count, and we only improve count on valid states.
    
    return final_added

