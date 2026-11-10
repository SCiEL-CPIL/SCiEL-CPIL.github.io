import random
import time
from itertools import combinations

def construct():
    n = 18
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_edge_in_mask(mask, adj):
        # Check if induced subgraph on 'mask' has at least one edge
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            u = (lsb.bit_length() - 1)
            # Check if u has any neighbor in mask (other than itself)
            # adj[u] contains neighbors.
            # We need to check if adj[u] & mask has any bits other than u.
            # Since adj[u] doesn't contain u (simple graph), adj[u] & mask is fine.
            if adj[u] & mask:
                return True
            m -= lsb
        return False

    def is_k4_safe(adj, u, v):
        # Check if adding edge (u,v) creates a K4
        # Common neighbors C = adj[u] & adj[v]
        c = adj[u] & adj[v]
        # If C has any edge, K4 is formed
        return not has_edge_in_mask(c, adj)

    def find_triangle_free_set(adj, size, max_depth=18):
        # Returns a bitmask of a triangle-free set of given size, or None
        # Uses backtracking
        # To make it faster, we can use a heuristic order
        vertices = list(range(n))
        
        def backtrack(remaining_mask, current_mask, current_size):
            if current_size == size:
                return current_mask
            if current_size + popcount(remaining_mask) < size:
                return None
            
            # Heuristic: pick a vertex with high degree in remaining? 
            # Or just pick the first one.
            # Let's pick the first available vertex to keep it simple and deterministic-ish
            # Actually, randomizing the order might help escape local minima in the "breaking" phase, 
            # but here we just want *a* set.
            
            m = remaining_mask
            if m == 0:
                return None
            
            # Get first vertex
            lsb = m & -m
            v = lsb.bit_length() - 1
            v_bit = 1 << v
            
            # Option 1: Include v
            # Check if adding v creates a triangle with current_mask
            # Triangle formed if there is an edge (a,b) in current_mask 
            # such that v is connected to both a and b.
            # This is equivalent to: The neighborhood of v within current_mask 
            # contains an edge.
            neighbors_in_current = adj[v] & current_mask
            if not has_edge_in_mask(neighbors_in_current, adj):
                # Safe to add
                res = backtrack(remaining_mask ^ v_bit, current_mask | v_bit, current_size + 1)
                if res is not None:
                    return res
            
            # Option 2: Exclude v
            res = backtrack(remaining_mask ^ v_bit, current_mask, current_size)
            if res is not None:
                return res
                
            return None

        full_mask = (1 << n) - 1
        return backtrack(full_mask, 0, 0)

    def get_max_tf_size(adj):
        # Binary search or linear search from high to low
        for s in range(n, 0, -1):
            if find_triangle_free_set(adj, s) is not None:
                return s
        return 0

    best_edges = None
    best_score = n # 18

    # Try multiple random instances
    for trial in range(500):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(trial)
        adj = [0] * n
        edges = set()
        
        # Generate random K4-free graph
        # Shuffle all pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_k4_safe(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        # Try to improve by breaking triangle-free sets of size 12
        # We want to reduce the max size.
        # If we can get it down to 11, great.
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Check if there is a triangle-free set of size 12
            # If yes, try to break it
            # If we break all 12s, check 11s? 
            # The score is the MAX size. So if max is 12, score is 12.
            # We want to minimize this.
            
            target_size = 12
            # Check if max tf size is > 11
            # Actually, let's just find the max size first? 
            # No, that's expensive.
            # Just try to break size 12 sets.
            
            tf_set_mask = find_triangle_free_set(adj, target_size)
            if tf_set_mask is not None:
                # Found a set of 12 vertices with no triangles
                # Try to add an edge within this set to break it
                # Get list of vertices in set
                verts_in_set = [i for i in range(n) if (tf_set_mask >> i) & 1]
                
                # Try all pairs in this set
                added = False
                random.shuffle(verts_in_set) # Randomize order to avoid bias
                
                for i in range(len(verts_in_set)):
                    for j in range(i+1, len(verts_in_set)):
                        u = verts_in_set[i]
                        v = verts_in_set[j]
                        
                        # Check if edge already exists
                        if (adj[u] >> v) & 1:
                            continue
                            
                        # Check K4 safety
                        if is_k4_safe(adj, u, v):
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.add((u, v))
                            added = True
                            break
                    if added:
                        break
                
                if added:
                    improved = True
                else:
                    # Could not add any edge to this specific set.
                    # Maybe try another set?
                    # The find function returns the first one it finds.
                    # To find another, we'd need to modify the search.
                    # For now, if we can't break *a* set, we might be stuck.
                    # Let's restart with a new random graph if we get stuck here.
                    # Or, we can try to break size 11 sets if we are already at 11?
                    # But if we have a size 12 set, we MUST break it to get score < 12.
                    # If we can't break it, this graph is stuck at >= 12.
                    pass
                    
            # If we successfully broke all size 12 sets, find_triangle_free_set(adj, 12) will return None.
            # Then the loop exits.
            # We should then check if we can break size 11 sets to get score 10?
            # The task says "Minimize".
            # So we should continue trying to break 11s if 12s are gone.
            
            # Let's generalize the loop:
            # Find max S such that tf set of size S exists.
            # If S > 11, try to break S.
            # If S <= 11, we are good? 
            # Actually, if S=11, score is 11. Can we get 10?
            # Let's try to break 11s as well.
            
            # Revised loop logic:
            # 1. Find max S.
            # 2. If S > 10:
            #    Find a set of size S.
            #    Try to break it.
            #    If fail, break loop (stuck).
            # 3. If S <= 10, break loop (optimal enough?).
            
            # But finding max S is expensive.
            # Let's just iterate target_size from 12 down to 10.
            
            # Wait, the current `improved` logic only checks 12.
            # Let's change the inner loop to check 12, then 11, then 10.
            
            pass # Placeholder for restructure

        # After the while loop, evaluate the final graph
        # Calculate exact max tf size
        final_score = get_max_tf_size(adj)
        
        if final_score < best_score:
            best_score = final_score
            best_edges = list(edges)
            
            if best_score <= 10: # Good enough?
                break

    if best_edges is None:
        # Fallback to a simple random graph
        random.seed(42)
        adj = [0] * n
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_k4_safe(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        best_edges = list(edges)

    return [[u, v] for u, v in best_edges]
