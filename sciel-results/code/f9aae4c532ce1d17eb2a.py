import time
import random

def construct():
    N = 20
    deadline = time.time() + 100 # Safety margin
    best_edges = []
    best_count = 0
    
    # Initialize with a random valid graph or empty
    # Let's start with a random graph and repair it, or just start empty and add.
    # Starting empty and adding is safer to ensure validity.
    
    def solve_one_run():
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * N
        edges = []
        
        # Helper to check if adding (u, v) is valid
        def can_add(u, v):
            if u == v: return False
            if (adj[u] >> v) & 1: return True # Already added
            
            # 1. Check pair (u, v)
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                return False
            
            # 2. Check impact on pairs (u, w) for w in N(v)
            # If w is in N(v), then v becomes a common neighbor of u and w.
            # So we need |N(u) \ {v} \cap N(w) \ {v}| + 1 <= 2  => |N(u) \cap N(w)| <= 2 (since v not in N(u) yet)
            # Wait, v is NOT in N(u) yet.
            # New common neighbors of (u, w) = (N(u) \cap N(w)) U {v} (if v in N(w)).
            # Since v in N(w), v is in the new set.
            # Is v in N(u) \cap N(w)? No, v not in N(u).
            # So new count = old_count(u, w) + 1.
            # We need new_count <= 2 => old_count <= 1.
            
            # So for all w in N(v):
            #   if (adj[u] & adj[w]).bit_count() > 1: return False
            # Note: adj[u] does not contain v.
            
            for w in range(N):
                if (adj[v] >> w) & 1:
                    if (adj[u] & adj[w]).bit_count() > 1:
                        return False
                        # Wait, is this correct?
                        # Common neighbors of u and w:
                        # Old: N(u) \cap N(w)
                        # New: N(u) \cap N(w) U {v} (because v is neighbor of u now? No, we are CHECKING if we CAN add. If we add, v becomes neighbor of u.)
                        # If we add (u,v), then N'(u) = N(u) U {v}.
                        # Common(u, w) becomes (N(u) U {v}) \cap N(w).
                        # Since w is in N(v), v is in N(w).
                        # So v is in the intersection.
                        # Is v in N(u)? No.
                        # So the new intersection is (N(u) \cap N(w)) U {v}.
                        # Size is |N(u) \cap N(w)| + 1.
                        # We need this <= 2.
                        # So |N(u) \cap N(w)| <= 1.
                        # Correct.
                        
            # 3. Check impact on pairs (v, w) for w in N(u)
            for w in range(N):
                if (adj[u] >> w) & 1:
                    if (adj[v] & adj[w]).bit_count() > 1:
                        return False
            
            return True

        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))

        # Greedy Phase
        # Randomize order of edges
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
                
        # Local Search Phase
        # Try to add missing edges. If stuck, remove one and retry.
        while time.time() < deadline:
            # Find valid missing edges
            valid_missing = []
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add(u, v):
                        valid_missing.append((u, v))
                        
            if valid_missing:
                # Add a random valid missing edge
                u, v = random.choice(valid_missing)
                add_edge(u, v)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
            else:
                # Stuck. Remove a random edge to kick
                if edges:
                    u, v = random.choice(edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges.remove((u, v))
                    # Optional: try to add some edges immediately after removal
                    # But the loop will handle it.
                    
        return best_edges, best_count

    # Run multiple times?
    # The greedy phase is fast. Local search is the bottleneck.
    # We can run the whole process multiple times if time permits.
    
    # Since the deadline is 100s, and one run might take a while,
    # let's just run one long local search with kicks.
    # Or run multiple short runs and take the best.
    # Given N=20, the search space is small enough that one good run might suffice,
    # but random restarts help.
    
    # Let's do a loop of restarts.
    while time.time() < deadline:
        edges, count = solve_one_run()
        # Update global best
        # Actually solve_one_run updates its local best, but we need to return the best found.
        # Let's refactor to return the best from that run.
        
    # Wait, the current structure of solve_one_run returns the best *within* that run.
    # We should keep a global best.
    
    # Let's rewrite to be cleaner.
