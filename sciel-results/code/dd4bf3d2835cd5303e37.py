"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    n = 20
    
    # Precompute pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    pair_index = {p: i for i, p in enumerate(pairs)}
    
    # Bitmask representation for adjacency
    # adj[u] is an integer where bit v is 1 if (u,v) is an edge
    adj = [0] * n
    
    def count_common_neighbors(u, v):
        """Count common neighbors of u and v using bitmasks."""
        # Remove u and v from the intersection to ensure we only count common neighbors
        # Actually, adj[u] & adj[v] contains bits for all common neighbors.
        # Since the graph is simple, u is not in adj[u] and v is not in adj[v].
        # So adj[u] & adj[v] is exactly the set of common neighbors.
        return bin(adj[u] & adj[v]).count('1')
        
    def is_k23_free():
        """Check if graph is K2,3-free: no two vertices have 3+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(u, v) >= 3:
                    return False
        return True
        
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        return sum(bin(a).count('1') for a in adj) // 2

    best_edges = []
    best_count = 0
    
    # Strategy: 
    # 1. Try several random graphs G(n, p) with p tuned to be dense but K2,3-free.
    #    For K2,3, the extremal number ex(n, K2,3) is roughly O(n^{3/2}).
    #    For n=20, n^{3/2} ~ 89. But constant factors matter.
    #    A known construction for K2,3-free graphs involves polarity graphs or similar.
    #    Let's try p in [0.1, 0.3].
    
    # Also try a greedy construction from scratch as a baseline for local search.
    
    # Random seed for reproducibility in this run, but we want variety.
    # We will run multiple restarts.
    
    num_restarts = 20
    steps_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize graph
        # Mix of random and greedy
        if r % 2 == 0:
            # Random initialization
            p = 0.15 + 0.05 * (r // 2) # Vary p
            for u, v in pairs:
                if random.random() < p:
                    add_edge(u, v)
        else:
            # Greedy initialization: add edges if they don't create K2,3
            # Sort edges by random order to break symmetry
            shuffled_pairs = pairs[:]
            random.shuffle(shuffled_pairs)
            for u, v in shuffled_pairs:
                # Check if adding (u,v) creates a K2,3
                # Adding (u,v) affects common neighbors of (u, x) for x in N(v) 
                # and (v, x) for x in N(u)
                # And common neighbors of (u,v) themselves? No, (u,v) are not neighbors of themselves.
                # The condition is: for any pair (a,b), common_neighbors(a,b) < 3.
                # Adding (u,v) increases common_neighbors(u,v) by 0 (they are now neighbors, but common neighbors are other nodes).
                # Wait, common neighbors of u and v are nodes w such that (u,w) and (v,w) exist.
                # Adding (u,v) does NOT change the set of common neighbors of u and v.
                # It DOES change common neighbors of (u, w) for w in N(v) \ {u}.
                # And (v, w) for w in N(u) \ {v}.
                
                # Let's just check all pairs affected.
                # Affected pairs: (u, x) where x is neighbor of v.
                #              (v, x) where x is neighbor of u.
                #              (u, v) itself? No, common neighbors of u and v don't change.
                
                valid = True
                # Check pairs involving u
                for x in range(n):
                    if x == u or x == v: continue
                    if adj[v] & (1 << x): # x is neighbor of v
                        # Adding (u,v) makes u and x share v as a common neighbor?
                        # No, common neighbors of u and x are nodes y s.t. (u,y) and (x,y).
                        # If we add (u,v), does it affect common(u,x)?
                        # Yes, if v is a neighbor of x, then v becomes a common neighbor of u and x.
                        # So we need to check if common(u, x) + (1 if v in N(x) else 0) >= 3
                        # But v is in N(x) is equivalent to x in N(v).
                        # So for each x in N(v), we check common(u, x).
                        # If common(u, x) is currently 2, and v is in N(x) (which it is), then adding (u,v) makes common(u,x) become 3. Invalid.
                        c = count_common_neighbors(u, x)
                        if c >= 2: # If it's 2, adding edge makes it 3? 
                            # Wait. count_common_neighbors(u, x) counts y such that (u,y) and (x,y).
                            # If we add (u,v), does v become a common neighbor?
                            # v is a common neighbor of u and x if (u,v) and (x,v) exist.
                            # We are adding (u,v). (x,v) exists because x in N(v).
                            # So yes, v becomes a new common neighbor.
                            # So if current common(u,x) is 2, new common is 3. Invalid.
                            valid = False
                            break
                if not valid:
                    continue
                    
                # Check pairs involving v
                for x in range(n):
                    if x == u or x == v: continue
                    if adj[u] & (1 << x): # x is neighbor of u
                        c = count_common_neighbors(v, x)
                        if c >= 2:
                            valid = False
                            break
                if not valid:
                    continue
                
                add_edge(u, v)
        
        # Local Search (Simulated Annealing / Hill Climbing)
        current_edges = edge_count()
        
        # Update best
        if current_edges > best_count:
            best_count = current_edges
            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u] & (1 << v)]
            
        # Local search steps
        # We will try to add random edges or swap edges.
        # Since we want to maximize edges, adding is better.
        # If we can't add, we might need to remove some to allow others? 
        # For now, just try adding random missing edges.
        
        missing_edges = [(u, v) for u, v in pairs if not (adj[u] & (1 << v))]
        
        for _ in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break
                
            if not missing_edges:
                break
                
            # Pick a random missing edge
            u, v = random.choice(missing_edges)
            
            # Check if adding (u,v) is valid
            # Same logic as greedy
            valid = True
            for x in range(n):
                if x == u or x == v: continue
                if adj[v] & (1 << x):
                    if count_common_neighbors(u, x) >= 2:
                        valid = False
                        break
            if valid:
                for x in range(n):
                    if x == u or x == v: continue
                    if adj[u] & (1 << x):
                        if count_common_neighbors(v, x) >= 2:
                            valid = False
                            break
            
            if valid:
                add_edge(u, v)
                current_edges += 1
                # Remove from missing list (lazy deletion: just check existence in list? 
                # Better to rebuild or use a set. For n=20, list of ~100-150 is small.
                # To avoid O(N) removal, we can just filter later or use a set.
                # Let's use a set for missing edges for O(1) removal.
                pass # We'll switch to set if needed, but list is fine for 190 items.
                
                # Update best
                if current_edges > best_count:
                    best_count = current_edges
                    best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u] & (1 << v)]
                    
                # Remove (u,v) from missing_edges
                try:
                    missing_edges.remove((u, v))
                except ValueError:
                    pass
            else:
                # Optionally, try to remove an edge to allow future additions?
                # This is complex. Stick to greedy addition for now.
                # If stuck, the next restart might find a different path.
                pass
                
    # Final check and return
    if best_count == 0:
        # Fallback to seed if something went wrong
        return [[4 * k + a, 4 * k + b] for k in range(5) for a in range(4) for b in range(a + 1, 4)]
        
    return best_edges

# EVOLVE-BLOCK-END
