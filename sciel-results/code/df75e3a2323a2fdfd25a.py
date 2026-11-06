import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    n = 16
    target_edges = 65
    
    # Initialize with the seed structure
    # K_{8,8} + edge (0,1)
    # Vertices 0-7 are part A, 8-15 are part B
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Add K_{8,8} edges
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append([u, v])
            
    # Add the extra edge (0, 1)
    adj[0][1] = 1
    adj[1][0] = 1
    edges.append([0, 1])
    
    # Degrees
    deg = [0]*n
    for u in range(n):
        deg[u] = sum(adj[u])
        
    # Helper to compute score
    def get_score(adj, deg):
        max_sum = 0
        # Iterate over all potential triangles
        # u < v < w
        for u in range(n):
            if deg[u] == 0: continue
            for v in range(u+1, n):
                if not adj[u][v]: continue
                if deg[v] == 0: continue
                # Check common neighbors w > v
                # We can iterate w from v+1 to n-1
                for w in range(v+1, n):
                    if adj[v][w] and adj[u][w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    current_score = get_score(adj, deg)
    best_score = current_score
    best_adj = [row[:] for row in adj]
    
    # Local search
    # Move: swap an edge (u,v) with a non-edge (x,y)
    # To keep edge count constant.
    
    # Precompute list of edges and non-edges for faster sampling
    edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edge_set.add((u,v))
                
    all_pairs = [(u,v) for u in range(n) for v in range(u+1, n)]
    non_edge_set = set(all_pairs) - edge_set
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    
    # Track elapsed time
    while time.time() - start_time < timeout:
        if not non_edge_set:
            break
            
        # Pick a random edge and non-edge
        u, v = random.choice(list(edge_set))
        x, y = random.choice(list(non_edge_set))
        
        # Check if swap is valid (simple graph)
        # We remove (u,v) and add (x,y)
        # Ensure no self loops (already handled by pairs)
        
        # Tentative change
        # Remove u-v
        adj[u][v] = 0
        adj[v][u] = 0
        deg[u] -= 1
        deg[v] -= 1
        
        # Add x-y
        adj[x][y] = 1
        adj[y][x] = 1
        deg[x] += 1
        deg[y] += 1
        
        new_score = get_score(adj, deg)
        
        # Accept if better or with probability exp(-(new-old)/T)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                # Optional: restart if we hit a very good score? 
                # For now, just keep going.
        else:
            # Revert
            adj[u][v] = 1
            adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
            
            adj[x][y] = 0
            adj[y][x] = 0
            deg[x] -= 1
            deg[y] -= 1
            
            # Acceptance probability
            delta = new_score - current_score
            if delta > 0:
                p = np.exp(-delta / T)
                if random.random() < p:
                    # Accept worse move
                    adj[u][v] = 0
                    adj[v][u] = 0
                    deg[u] -= 1
                    deg[v] -= 1
                    
                    adj[x][y] = 1
                    adj[y][x] = 1
                    deg[x] += 1
                    deg[y] += 1
                    current_score = new_score
                # else revert (already reverted)
            else:
                # This shouldn't happen if new_score >= current_score check above
                # But if equal, accept?
                adj[u][v] = 0
                adj[v][u] = 0
                deg[u] -= 1
                deg[v] -= 1
                
                adj[x][y] = 1
                adj[y][x] = 1
                deg[x] += 1
                deg[y] += 1
                current_score = new_score

        T *= cooling
        if T < 0.1:
            T = 0.1 # Floor temperature to keep exploring
            
        # Update sets if accepted
        # We need to update edge_set and non_edge_set if the move was accepted
        # This is inefficient if we do it every step. 
        # Optimization: Only update sets when a move is accepted.
        # But we need to know if it was accepted.
        
        # Let's restructure to track acceptance clearly.
        # Actually, the logic above is a bit messy with reverts.
        # Let's use a cleaner swap function.
        
        # Restart loop logic for clarity and correctness of state tracking
        # The above loop modifies adj/deg in place. If rejected, it reverts.
        # If accepted, we must update edge_set/non_edge_set.
        
        # To avoid bugs, let's just rebuild sets periodically or use a different strategy.
        # Given n=16, rebuilding sets is O(n^2) = 256 ops, which is fast.
        # But doing it every step might be slow if steps are many.
        # Let's only rebuild if accepted.
        
        # Wait, the code above doesn't update edge_set/non_edge_set.
        # This means the sampler is using stale sets. 
        # It might pick an edge that is no longer an edge, or a non-edge that is now an edge.
        # We must update the sets.
        
        # Let's rewrite the inner loop to handle state updates correctly.
        
        # Break to implement cleaner version below if this gets complex.
        # The current implementation is flawed regarding set updates.
        
        # Let's restart the search loop with proper state management.
        break # Exit this flawed loop structure

    # --- Clean Implementation ---
    
    # Re-initialize to be safe, or just continue from best_adj found?
    # The previous loop might have found something, but the state is messy.
    # Let's run a proper search from the best_adj found or the seed.
    
    # Use best_adj from the first (flawed) pass if it improved, otherwise seed.
    # Actually, let's just run a robust search from the start or best known.
    
    # Let's use the best_adj from the first pass if valid, else seed.
    # But the first pass logic was broken. Let's discard it and start fresh with a correct implementation.
    
    # Start from Seed
    adj = [[0]*n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
    adj[0][1] = 1
    adj[1][0] = 1
    
    deg = [0]*n
    for u in range(n):
        deg[u] = sum(adj[u])
        
    current_score = get_score(adj, deg)
    best_score = current_score
    best_adj = [row[:] for row in adj]
    
    # Build sets
    edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edge_set.add((u,v))
    all_pairs = [(u,v) for u in range(n) for v in range(u+1, n)]
    non_edge_set = set(all_pairs) - edge_set
    
    T = 2.0
    cooling = 0.9995
    
    # Main Search Loop
    while time.time() - start_time < timeout:
        if not non_edge_set:
            break
            
        # Random swap
        u, v = random.choice(list(edge_set))
        x, y = random.choice(list(non_edge_set))
        
        # Apply swap
        adj[u][v] = 0
        adj[v][u] = 0
        deg[u] -= 1
        deg[v] -= 1
        
        adj[x][y] = 1
        adj[y][x] = 1
        deg[x] += 1
        deg[y] += 1
        
        new_score = get_score(adj, deg)
        
        accepted = False
        if new_score < current_score:
            accepted = True
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
        else:
            # Revert
            adj[u][v] = 1
            adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
            
            adj[x][y] = 0
            adj[y][x] = 0
            deg[x] -= 1
            deg[y] -= 1
            
            delta = new_score - current_score
            if delta > 0:
                p = np.exp(-delta / T)
                if random.random() < p:
                    # Accept worse
                    adj[u][v] = 0
                    adj[v][u] = 0
                    deg[u] -= 1
                    deg[v] -= 1
                    
                    adj[x][y] = 1
                    adj[y][x] = 1
                    deg[x] += 1
                    deg[y] += 1
                    current_score = new_score
                    accepted = True
            else:
                # Equal score, accept to explore
                adj[u][v] = 0
                adj[v][u] = 0
                deg[u] -= 1
                deg[v] -= 1
                
                adj[x][y] = 1
                adj[y][x] = 1
                deg[x] += 1
                deg[y] += 1
                current_score = new_score
                accepted = True

        if accepted:
            # Update sets
            edge_set.discard((u, v))
            non_edge_set.add((u, v))
            edge_set.add((x, y))
            non_edge_set.discard((x, y))
            
        T *= cooling
        if T < 0.05:
            T = 0.05

    # Convert best_adj to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                best_edges.append([u, v])
                
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
