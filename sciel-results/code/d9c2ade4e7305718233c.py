import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    D = 11

    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Precompute conflict masks using Python integers (bitsets)
    # conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * N_S
    for i in range(N_S):
        s_i = supports[i]
        for j in range(i + 1, N_S):
            s_j = supports[j]
            # Check intersection size
            # Since supports are tuples of sorted ints, we can use set intersection
            # But for speed, we can precompute masks for coordinates
            # Let's use coordinate masks
            pass
            
    # Better: Precompute coordinate masks for each support
    # coord_mask[i] is an int with bits set for coordinates in support i
    coord_mask = [0] * N_S
    for i, s in enumerate(supports):
        m = 0
        for idx in s:
            m |= (1 << idx)
        coord_mask[i] = m

    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = coord_mask[i] & coord_mask[j]
            if inter.bit_count() >= 3:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    ALL_ONES = (1 << N_S) - 1
    
    def greedy_fill(selected_mask, blocked_mask):
        """
        Greedily add nodes to selected_mask that are not in blocked_mask.
        Returns updated selected_mask, blocked_mask.
        """
        available = ALL_ONES & ~(selected_mask | blocked_mask)
        while available:
            # Pick a node. Strategy: pick one with fewest conflicts in available?
            # For speed, just pick the lowest set bit (or random)
            # Let's pick random to allow restarts
            # But for greedy fill, deterministic is often better for stability
            # We'll pick the one with min degree in the *remaining* graph?
            # That's expensive. Let's just pick the lowest index for now, 
            # or we can pass a priority list.
            
            # Optimization: pick node with min conflicts among available
            # This is O(N) per step. N=330. Steps ~15. Total ~5000 ops. Fast.
            min_deg = float('inf')
            best_v = -1
            temp_avail = available
            while temp_avail:
                v = (temp_avail & -temp_avail).bit_length() - 1
                temp_avail &= temp_avail - 1
                # Count conflicts in available
                # neighbors = conflict_mask[v] & available
                # deg = neighbors.bit_count()
                # This is fast with bit_count
                deg = (conflict_mask[v] & available).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if min_deg == 0:
                        break
                        
            if best_v == -1:
                break
                
            selected_mask |= (1 << best_v)
            blocked_mask |= conflict_mask[best_v]
            available = ALL_ONES & ~(selected_mask | blocked_mask)
            
        return selected_mask, blocked_mask

    def compute_blocked(selected_mask):
        """Compute blocked mask from selected mask."""
        blocked = 0
        temp = selected_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            blocked |= conflict_mask[v]
        return blocked

    def local_search(selected_mask, time_limit):
        """
        Local search using 'remove one, greedy fill' moves.
        """
        best_mask = selected_mask
        best_size = best_mask.bit_count()
        blocked_mask = compute_blocked(best_mask)
        
        # Iterate
        while time.time() < time_limit:
            improved = False
            # Try removing each node in best_mask
            temp_sel = best_mask
            while temp_sel:
                u = (temp_sel & -temp_sel).bit_length() - 1
                temp_sel &= temp_sel - 1
                
                # Remove u
                new_sel = best_mask & ~(1 << u)
                new_blocked = compute_blocked(new_sel)
                
                # Greedy fill on new state
                new_sel, new_blocked = greedy_fill(new_sel, new_blocked)
                
                new_size = new_sel.bit_count()
                if new_size > best_size:
                    best_mask = new_sel
                    best_size = new_size
                    blocked_mask = new_blocked
                    improved = True
                    # Restart the loop over nodes with new best_mask
                    temp_sel = best_mask
                    break
            
            if not improved:
                # If no improvement found in one full pass, break or do a kick
                # To avoid getting stuck, we can do a random kick
                # Remove 2 random nodes, greedy fill
                if best_size >= 2:
                    bits = []
                    t = best_mask
                    while t:
                        v = (t & -t).bit_length() - 1
                        bits.append(v)
                        t &= t - 1
                    if len(bits) >= 2:
                        idx1, idx2 = random.sample(range(len(bits)), 2)
                        u1, u2 = bits[idx1], bits[idx2]
                        new_sel = best_mask & ~(1 << u1) & ~(1 << u2)
                        new_blocked = compute_blocked(new_sel)
                        new_sel, new_blocked = greedy_fill(new_sel, new_blocked)
                        new_size = new_sel.bit_count()
                        if new_size > best_size:
                            best_mask = new_sel
                            best_size = new_size
                            blocked_mask = new_blocked
                            improved = True
                            
                if not improved:
                    break
                    
        return best_mask

    best_mask = 0
    best_size = 0
    
    # Multi-start
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        # Start with empty set and greedy fill
        # Or start with a random subset and greedy fill?
        # Greedy fill from empty is a good baseline.
        # To diversify, we can shuffle the "priority" in greedy fill.
        # My greedy_fill picks min-degree. 
        # Let's add randomness to the selection in greedy_fill?
        # Or just run greedy_fill multiple times?
        
        # Strategy: Run greedy_fill with random tie-breaking or random order?
        # The current greedy_fill picks min-degree. 
        # Let's modify greedy_fill to accept a random seed or just rely on the order of bits?
        # The order of bits is fixed (0 to N-1).
        # To diversify, we can permute the node indices? That requires remapping conflict masks. Expensive.
        
        # Alternative: Start with a random valid set?
        # Pick random nodes, ensure no conflict, then greedy fill.
        
        init_sel = 0
        init_blocked = 0
        if r % 2 == 0:
            # Pure greedy
            init_sel, init_blocked = greedy_fill(0, 0)
        else:
            # Random seed
            # Pick 5 random non-conflicting nodes
            candidates = list(range(N_S))
            random.shuffle(candidates)
            for v in candidates:
                if not (init_sel >> v) & 1 and not (init_blocked >> v) & 1:
                    init_sel |= (1 << v)
                    init_blocked |= conflict_mask[v]
            init_sel, init_blocked = greedy_fill(init_sel, init_blocked)
            
        # Local search
        remaining = deadline - time.time() - 2.0
        if remaining < 0.5:
            continue
            
        res_mask = local_search(init_sel, deadline - 2.0)
        res_size = res_mask.bit_count()
        
        if res_size > best_size:
            best_size = res_size
            best_mask = res_mask
            
        # Early stop if we reach a theoretical bound?
        # Max possible K is around 15-20.
        # If best_size > 18, it's very good.
        if best_size >= 20:
            break

    # Build vectors
    vectors = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Extract supports from best_mask
    temp = best_mask
    while temp:
        idx = (temp & -temp).bit_length() - 1
        temp &= temp - 1
        support = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for coord, sign in zip(support, signs):
                v[coord] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return np.array(vectors, dtype=np.int8)
