import time
import sys
import random

DEADLINE_MARGIN = 15.0
HARD_LIMIT = 135.0

def _is_3_colorable_bitmask(adj_mask, n):
    """
    Check if graph with adjacency bitmasks is 3-colorable.
    adj_mask is a list of n integers, where bit i is 1 if vertex i is a neighbor.
    """
    # Order vertices by degree (popcount) descending
    order = sorted(range(n), key=lambda x: -bin(adj_mask[x]).count('1'))
    
    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Determine forbidden colors from already colored neighbors
        forbidden = 0
        mask = adj_mask[v]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            nb_idx = lsb.bit_length() - 1
            c = colors[nb_idx]
            if c != -1:
                forbidden |= (1 << c)
            m ^= lsb
        
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        
        return False
    
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Initialize all pairs as open
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            
    # For B&B, we maintain forced and open masks
    # forced[i] bitmask of neighbors in forced
    # open[i] bitmask of neighbors in open
    
    def init_masks():
        forced = [0] * n
        open_mask = [0] * n
        for u, v in all_edges:
            open_mask[u] |= (1 << v)
            open_mask[v] |= (1 << u)
        return forced, open_mask

    best_count = 0
    best_edges = []
    
    # We will run multiple B&B searches with different random orderings
    random.seed(42)
    
    # Precompute edge list for ordering
    # We want to try to find a graph with many edges.
    
    def count_edges(mask_list):
        return sum(bin(m).count('1') for m in mask_list) // 2

    def get_edges_from_mask(mask_list):
        edges = []
        for u in range(n):
            m = mask_list[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # Branch and Bound
    def solve(forced, open_mask, current_count, forced_non_3c):
        nonlocal best_count, best_edges, deadline
        
        if time.time() > deadline:
            return

        # Pruning 1: Upper bound
        # Max possible edges = current_count + count_edges(open_mask)
        open_count = count_edges(open_mask)
        upper_bound = current_count + open_count
        
        if upper_bound <= best_count:
            return

        # Pruning 2: 3-Colorability
        # If forced is already non-3-colorable, we don't need to check colorability of forced+open
        # because any superset is also non-3-colorable.
        if not forced_non_3c:
            # Check if forced + open is 3-colorable
            # Construct combined mask
            combined = [forced[i] | open_mask[i] for i in range(n)]
            if _is_3_colorable_bitmask(combined, n):
                return # Cannot achieve non-3-colorable in this branch
        
        # If open is empty, we have a candidate
        if open_count == 0:
            if current_count > best_count:
                best_count = current_count
                best_edges = get_edges_from_mask(forced)
            return

        # Select an edge from open
        # Heuristic: Pick an edge that is "constrained" or random?
        # Let's pick the first available edge in a shuffled order
        # To allow restarts, we pass a shuffled list of open edges?
        # Or just pick one deterministically based on index, and use restarts with different initial shuffles?
        # Better: Pick the edge with the most conflicts (highest degree in conflict graph) to prune faster?
        # For simplicity, let's pick a random edge from open, or the first one.
        # Given the small size, picking the first one in a fixed order might get stuck.
        # Let's pick the edge (u,v) from open that minimizes the number of remaining options?
        # Or just pick the first one.
        
        # Find first open edge
        u = -1
        v = -1
        for i in range(n):
            m = open_mask[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    u = i
                    v = j
                    break
                m ^= lsb
            if u != -1:
                break
        
        if u == -1:
            return # No open edges, handled above

        # Branch 1: Exclude (u,v)
        # Remove (u,v) from open
        open_mask[u] &= ~(1 << v)
        open_mask[v] &= ~(1 << u)
        solve(forced, open_mask, current_count, forced_non_3c)
        # Restore
        open_mask[u] |= (1 << v)
        open_mask[v] |= (1 << u)
        
        if time.time() > deadline:
            return

        # Branch 2: Include (u,v)
        # Check if adding (u,v) creates a triangle with forced
        # Triangle exists if forced[u] & forced[v] is not empty
        if forced[u] & forced[v]:
            return # Cannot add (u,v)

        # Add (u,v) to forced
        forced[u] |= (1 << v)
        forced[v] |= (1 << u)
        
        # Propagate: Remove edges from open that would form triangles with (u,v) and forced
        # If (u,w) is in forced, then (v,w) cannot be in open
        # If (v,w) is in forced, then (u,w) cannot be in open
        
        # Note: forced[u] now includes v.
        # We need to iterate over neighbors of u in forced (including v? No, v is the new edge)
        # Actually, if (u,w) is in forced, (v,w) is forbidden.
        # w can be any vertex.
        
        # Get neighbors of u in forced BEFORE adding v?
        # It doesn't matter much, just ensure we don't process v twice or incorrectly.
        # Let's use the updated forced.
        
        mu = forced[u]
        while mu:
            lsb = mu & -mu
            w = lsb.bit_length() - 1
            if w != v: # w is a neighbor of u in forced
                # (v,w) cannot be added
                open_mask[v] &= ~(1 << w)
                open_mask[w] &= ~(1 << v)
            mu ^= lsb
            
        mv = forced[v]
        while mv:
            lsb = mv & -mv
            w = lsb.bit_length() - 1
            if w != u: # w is a neighbor of v in forced
                # (u,w) cannot be added
                open_mask[u] &= ~(1 << w)
                open_mask[w] &= ~(1 << u)
            mv ^= lsb
            
        # Remove (u,v) from open
        open_mask[u] &= ~(1 << v)
        open_mask[v] &= ~(1 << u)
        
        new_count = current_count + 1
        
        # Check if new forced is non-3-colorable
        new_forced_non_3c = forced_non_3c
        if not new_forced_non_3c:
            if not _is_3_colorable_bitmask(forced, n):
                new_forced_non_3c = True
                
        solve(forced, open_mask, new_count, new_forced_non_3c)
        
        # Restore
        # Restore open edges removed during propagation
        # This is tricky because we modified open_mask in place.
        # We should copy open_mask before modifying it for the branch.
        
    # The above implementation modifies open_mask in place and restoration is hard.
    # Better to pass copies or use a stack.
    # Given n=16, copying lists of 16 ints is cheap.
    
    def solve_copied(forced, open_mask, current_count, forced_non_3c):
        nonlocal best_count, best_edges, deadline
        
        if time.time() > deadline:
            return

        # Pruning 1: Upper bound
        open_count = count_edges(open_mask)
        upper_bound = current_count + open_count
        
        if upper_bound <= best_count:
            return

        # Pruning 2: 3-Colorability
        if not forced_non_3c:
            combined = [forced[i] | open_mask[i] for i in range(n)]
            if _is_3_colorable_bitmask(combined, n):
                return

        if open_count == 0:
            if current_count > best_count:
                best_count = current_count
                best_edges = get_edges_from_mask(forced)
            return

        # Find an open edge
        u = -1
        v = -1
        # Heuristic: Pick edge with highest sum of degrees in open?
        # Or just first.
        for i in range(n):
            m = open_mask[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    u = i
                    v = j
                    break
                m ^= lsb
            if u != -1:
                break
        
        if u == -1:
            return

        # Branch 1: Exclude (u,v)
        open_mask_excl = open_mask[:]
        open_mask_excl[u] &= ~(1 << v)
        open_mask_excl[v] &= ~(1 << u)
        solve_copied(forced, open_mask_excl, current_count, forced_non_3c)
        
        if time.time() > deadline:
            return

        # Branch 2: Include (u,v)
        if forced[u] & forced[v]:
            return

        forced_incl = forced[:]
        forced_incl[u] |= (1 << v)
        forced_incl[v] |= (1 << u)
        
        open_mask_incl = open_mask[:]
        
        # Propagate
        mu = forced_incl[u]
        while mu:
            lsb = mu & -mu
            w = lsb.bit_length() - 1
            if w != v:
                open_mask_incl[v] &= ~(1 << w)
                open_mask_incl[w] &= ~(1 << v)
            mu ^= lsb
            
        mv = forced_incl[v]
        while mv:
            lsb = mv & -mv
            w = lsb.bit_length() - 1
            if w != u:
                open_mask_incl[u] &= ~(1 << w)
                open_mask_incl[w] &= ~(1 << u)
            mv ^= lsb
            
        open_mask_incl[u] &= ~(1 << v)
        open_mask_incl[v] &= ~(1 << u)
        
        new_count = current_count + 1
        
        new_forced_non_3c = forced_non_3c
        if not new_forced_non_3c:
            if not _is_3_colorable_bitmask(forced_incl, n):
                new_forced_non_3c = True
                
        solve_copied(forced_incl, open_mask_incl, new_count, new_forced_non_3c)

    # Run B&B
    # Initial state
    initial_forced, initial_open = init_masks()
    
    # To improve search, we can try to seed the forced set with a known non-3-colorable core?
    # The Grötzsch graph is a good start.
    # But B&B from scratch should find it.
    # However, starting from empty forced might explore many 3-colorable branches.
    # We can add a heuristic: If we find a non-3-colorable core early, we update best_count.
    
    # Let's try a few random restarts with different edge orderings?
    # The current implementation picks the first edge.
    # We can shuffle the vertex indices to change the order?
    # Or just run it once. With good pruning, it might be fast enough.
    
    # To ensure we find a good solution, let's initialize best with Grötzsch graph edges
    # Grötzsch: C5 (0-1-2-3-4-0), Mycielski (5-9), Universal (10)
    groetzsch_edges = []
    for i in range(5):
        groetzsch_edges.append([i, (i + 1) % 5])
    for i in range(5):
        groetzsch_edges.append([5 + i, (i + 1) % 5])
        groetzsch_edges.append([5 + i, (i - 1) % 5])
    for i in range(5):
        groetzsch_edges.append([10, 5 + i])
        
    # Convert to masks
    best_count = len(groetzsch_edges)
    best_edges = groetzsch_edges
    
    forced_init, open_init = init_masks()
    
    # Mark Groetzsch edges as forced?
    # If we force them, we constrain the search.
    # But it gives a strong lower bound.
    # Let's just run B&B from scratch, but the initial best_count helps pruning.
    
    solve_copied(initial_forced, initial_open, 0, False)
    
    if not best_edges:
        # Fallback
        return groetzsch_edges
        
    return best_edges
