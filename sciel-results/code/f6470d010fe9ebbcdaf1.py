import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 108  # Safety margin before 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_bipartite(adj):
        """Check if graph is bipartite using BFS."""
        color = [-1] * n
        for i in range(n):
            if color[i] != -1:
                continue
            color[i] = 0
            queue = [i]
            while queue:
                u = queue.pop(0)
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
                    mask -= lsb
        return True

    def is_3_colorable(adj):
        """Check 3-colorability using backtracking with bitmasks and degree ordering."""
        # Quick check: if bipartite, it is 2-colorable, hence 3-colorable
        if is_bipartite(adj):
            return True
            
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Try each color
            for c in range(3):
                # Check if color c is available (no neighbor has color c)
                conflict = False
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    mask -= lsb
                
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    def edges_from_adj(adj):
        """Reconstruct edge list from adjacency bitmasks."""
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle. Returns True if added."""
        if (adj[u] >> v) & 1:
            return False
        # Check for common neighbor
        if (adj[u] & adj[v]) != 0:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    # Phase 1: Randomized maximal triangle-free graph search
    random.seed(42)
    
    while time.time() < deadline - 30:
        adj = [0] * n
        edge_count = 0
        
        # Shuffle edges for this trial
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Check if this maximal triangle-free graph is not 3-colorable
        if not is_3_colorable(adj):
            if edge_count > best_count:
                best_count = edge_count
                best_edges = edges_from_adj(adj)
                
    # Phase 2: Local Search with Edge Swaps (Remove then Greedy Add)
    if best_edges and time.time() < deadline - 5:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_count = best_count
        current_adj = adj[:]
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try removing each edge and see if we can get a better graph
            existing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (current_adj[u] >> v) & 1]
            random.shuffle(existing_edges)
            
            for u, v in existing_edges:
                if time.time() > deadline - 2:
                    break
                    
                # Remove edge (u,v)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Greedily add all possible edges to maximize edge count
                # We need to re-evaluate all missing edges
                # To be efficient, we can just iterate all edges and add if valid
                # But we must be careful not to add (u,v) back if it was the one we removed?
                # No, we want to see if we can get a *different* maximal graph with more edges.
                # Actually, if we remove an edge, the graph is no longer maximal.
                # We should try to fill it up.
                
                # Snapshot for rollback
                trial_adj = current_adj[:]
                
                # Add edges greedily in random order to encourage diversity?
                # Or just fixed order? Fixed order might be faster and deterministic.
                # Let's use a random order for the "add" phase to explore different maximal extensions.
                add_candidates = [(a, b) for a in range(n) for b in range(a+1, n) if not ((trial_adj[a] >> b) & 1)]
                random.shuffle(add_candidates)
                
                trial_count = current_count - 1 # We removed one
                for a, b in add_candidates:
                    if try_add_edge(trial_adj, a, b):
                        trial_count += 1
                
                # Check if the new graph is not 3-colorable
                if not is_3_colorable(trial_adj):
                    if trial_count > current_count:
                        current_count = trial_count
                        current_adj = trial_adj
                        best_count = current_count
                        best_edges = edges_from_adj(current_adj)
                        improved = True
                        break # Restart the outer loop with the new best
                    # If trial_count == current_count, we might still want to switch if it's different?
                    # But we want to maximize edges, so only switch if strictly better.
                # If trial_count < current_count, ignore.
                # If is_3_colorable, ignore.
                
                # Note: We don't revert current_adj explicitly because we only update it if improved.
                # But we did modify current_adj at the start of the loop iteration.
                # We must revert if we didn't improve.
                if not improved:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
    
    # Fallback: If we didn't find a good graph, use the 45-edge construction
    if best_count < 45:
        edges = []
        
        # 1. Grötzsch Graph Edges
        # Cycle on V (0-4)
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        
        # Cross edges between V and W
        for i in range(5):
            w_idx_1 = 5 + (i - 1) % 5
            w_idx_2 = 5 + (i + 1) % 5
            edges.append([i, w_idx_1])
            edges.append([i, w_idx_2])
        
        # Apex Z (10) connects to all W (5-9)
        for i in range(5):
            edges.append([10, 5 + i])
        
        # 2. Extension Edges
        # New vertices X (11-15) connect to all W (5-9)
        for x in range(11, 16):
            for w in range(5, 10):
                edges.append([x, w])
        
        # Total edges: 5 (cycle) + 10 (cross) + 5 (apex) + 25 (new) = 45
        return edges
    
    return best_edges
