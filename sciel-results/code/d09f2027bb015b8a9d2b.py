import random
import time
import math

N = 16
TOTAL_POINTS = N * N
DEADLINE = time.time() + 280  # Safety margin

def construct():
    # 1. Precompute isosceles masks
    # iso_mask[u][v] is a bitmask of points w such that (u, v, w) is isosceles
    # u, v are indices 0..255
    # w is distinct from u, v
    
    # Helper to get index
    def idx(x, y):
        return x * N + y
    
    # Precompute coordinates
    coords = [(i // N, i % N) for i in range(TOTAL_POINTS)]
    
    # Precompute squared distances
    def dist2(i, j):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        return (x1 - x2)**2 + (y1 - y2)**2
    
    # iso_mask[i][j]
    iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    
    for u in range(TOTAL_POINTS):
        for v in range(TOTAL_POINTS):
            if u == v:
                continue
            duv2 = dist2(u, v)
            mask = 0
            for w in range(TOTAL_POINTS):
                if w == u or w == v:
                    continue
                duw2 = dist2(u, w)
                dvw2 = dist2(v, w)
                if duv2 == duw2 or duv2 == dvw2 or duw2 == dvw2:
                    mask |= (1 << w)
            iso_mask[u][v] = mask
            # Symmetry
            # iso_mask[v][u] is same set of w, but we access by pair
            # Actually, the set of w is the same.
            # But we store it in [u][v]. We should ensure symmetry in access or fill both.
            # For efficiency, let's fill both.
            iso_mask[v][u] = mask

    # 2. SA Parameters
    LAMBDA = 100.0
    T_START = 10.0
    T_END = 0.1
    ALPHA = 0.995
    
    # State
    # S_mask: int
    # block_count: list of int
    # B_mask: int
    
    def popcount(x):
        return bin(x).count('1')
        
    def init_state():
        # Start with empty set
        S_mask = 0
        block_count = [0] * TOTAL_POINTS
        B_mask = 0
        return S_mask, block_count, B_mask

    def add_point(S_mask, block_count, B_mask, p):
        # Check validity: p not in S, p not in B
        if (S_mask & (1 << p)) != 0:
            return None
        if (B_mask & (1 << p)) != 0:
            return None
            
        new_S_mask = S_mask | (1 << p)
        
        # Update block_count and B_mask
        # For each u in S, update counts for w in iso_mask[p][u]
        new_block_count = block_count[:]  # Copy
        new_B_mask = B_mask
        
        # Iterate over set bits in S_mask
        # S is small, so we can just iterate 0..TOTAL_POINTS
        # But checking if bit is set is fast.
        for u in range(TOTAL_POINTS):
            if (S_mask & (1 << u)) != 0:
                mask_u = iso_mask[p][u]
                # Update counts for all w in mask_u
                # This is potentially slow if we do it naively.
                # But N=16, TOTAL_POINTS=256.
                # Max set bits in S is small.
                # Max bits in mask_u is small.
                
                # Efficient update:
                # We need to increment count for each w in mask_u.
                # And if count goes 0->1, set bit in B_mask.
                
                # Let's just iterate w
                w = 0
                m = mask_u
                while m:
                    if m & 1:
                        if new_block_count[w] == 0:
                            new_B_mask |= (1 << w)
                        new_block_count[w] += 1
                    m >>= 1
                    w += 1
        
        return new_S_mask, new_block_count, new_B_mask

    def remove_point(S_mask, block_count, B_mask, p):
        # p must be in S
        if (S_mask & (1 << p)) == 0:
            return None
            
        new_S_mask = S_mask & ~(1 << p)
        
        new_block_count = block_count[:]
        new_B_mask = B_mask
        
        for u in range(TOTAL_POINTS):
            if (new_S_mask & (1 << u)) != 0:
                mask_u = iso_mask[p][u]
                w = 0
                m = mask_u
                while m:
                    if m & 1:
                        new_block_count[w] -= 1
                        if new_block_count[w] == 0:
                            new_B_mask &= ~(1 << w)
                    m >>= 1
                    w += 1
        
        return new_S_mask, new_block_count, new_B_mask

    def energy(S_mask, B_mask):
        size_S = popcount(S_mask)
        size_B = popcount(B_mask)
        M = TOTAL_POINTS - size_S - size_B
        return size_S + LAMBDA * M

    # Start with a random valid independent set to avoid empty start issues?
    # Or start empty and let it grow?
    # Starting empty: E = 0 + LAMBDA * 256.
    # Adding a point: E changes.
    # Let's start with a greedy maximal set to have a good initial T and state.
    
    def greedy_maximal():
        S = []
        S_mask = 0
        block_count = [0] * TOTAL_POINTS
        B_mask = 0
        points = list(range(TOTAL_POINTS))
        random.shuffle(points)
        for p in points:
            if (S_mask & (1 << p)) == 0 and (B_mask & (1 << p)) == 0:
                # Add p
                S_mask |= (1 << p)
                for u in S:
                    mask_u = iso_mask[p][u]
                    w = 0
                    m = mask_u
                    while m:
                        if m & 1:
                            if block_count[w] == 0:
                                B_mask |= (1 << w)
                            block_count[w] += 1
                        m >>= 1
                        w += 1
                S.append(p)
        return S_mask, block_count, B_mask

    S_mask, block_count, B_mask = greedy_maximal()
    E_curr = energy(S_mask, B_mask)
    best_S_mask = S_mask
    best_E = E_curr
    
    T = T_START
    rng = random.Random(42)
    
    while time.time() < DEADLINE:
        # Choose move
        move_type = rng.random()
        if move_type < 0.5:
            # Add
            # Pick random p
            p = rng.randint(0, TOTAL_POINTS - 1)
            new_state = add_point(S_mask, block_count, B_mask, p)
            if new_state is None:
                continue
            new_S_mask, new_block_count, new_B_mask = new_state
            E_new = energy(new_S_mask, new_B_mask)
            dE = E_new - E_curr
            if dE < 0 or rng.random() < math.exp(-dE / T):
                S_mask, block_count, B_mask = new_S_mask, new_block_count, new_B_mask
                E_curr = E_new
                if E_curr < best_E:
                    best_E = E_curr
                    best_S_mask = S_mask
        else:
            # Remove
            # Pick random p in S
            if popcount(S_mask) == 0:
                continue
            # Find a set bit
            p = rng.randint(0, TOTAL_POINTS - 1)
            if (S_mask & (1 << p)) == 0:
                # Retry or pick from list
                # For efficiency, maintain list of points in S?
                # Or just retry a few times
                found = False
                for _ in range(10):
                    p = rng.randint(0, TOTAL_POINTS - 1)
                    if (S_mask & (1 << p)) != 0:
                        found = True
                        break
                if not found:
                    continue
            
            new_state = remove_point(S_mask, block_count, B_mask, p)
            if new_state is None:
                continue
            new_S_mask, new_block_count, new_B_mask = new_state
            E_new = energy(new_S_mask, new_B_mask)
            dE = E_new - E_curr
            if dE < 0 or rng.random() < math.exp(-dE / T):
                S_mask, block_count, B_mask = new_S_mask, new_block_count, new_B_mask
                E_curr = E_new
                if E_curr < best_E:
                    best_E = E_curr
                    best_S_mask = S_mask
        
        T *= ALPHA
        if T < T_END:
            T = T_END

    # Final cleanup: ensure best_S_mask is maximal
    # Convert mask to list
    best_points = [i for i in range(TOTAL_POINTS) if (best_S_mask & (1 << i)) != 0]
    
    # Greedily add any addable points
    # Reconstruct block state for best_S_mask
    # This is O(|S|^2 * K)
    block_count_final = [0] * TOTAL_POINTS
    B_mask_final = 0
    for i in range(len(best_points)):
        for j in range(i+1, len(best_points)):
            u = best_points[i]
            v = best_points[j]
            mask_uv = iso_mask[u][v]
            w = 0
            m = mask_uv
            while m:
                if m & 1:
                    if block_count_final[w] == 0:
                        B_mask_final |= (1 << w)
                    block_count_final[w] += 1
                m >>= 1
                w += 1
                
    # Addable points
    addable = []
    for p in range(TOTAL_POINTS):
        if (best_S_mask & (1 << p)) == 0 and (B_mask_final & (1 << p)) == 0:
            addable.append(p)
            
    # Add them greedily
    for p in addable:
        best_S_mask |= (1 << p)
        best_points.append(p)
        # Update block counts? Not strictly necessary if we just want the final set.
        # But we must ensure we don't add invalid points.
        # Since we checked addable, they are valid.
        # However, adding one might block others?
        # Yes. So we should re-check or just add one and recompute?
        # Given the small size, recomputing is safer.
        # But actually, if p is addable, it is not blocked by current S.
        # Adding p might block other addable points.
        # So we should iterate.
        
    # Let's just do a simple greedy loop on the final set
    final_set = list(best_points)
    final_mask = best_S_mask
    
    # Recompute B for final_set
    block_count_final = [0] * TOTAL_POINTS
    B_mask_final = 0
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            u = final_set[i]
            v = final_set[j]
            mask_uv = iso_mask[u][v]
            w = 0
            m = mask_uv
            while m:
                if m & 1:
                    if block_count_final[w] == 0:
                        B_mask_final |= (1 << w)
                    block_count_final[w] += 1
                m >>= 1
                w += 1
                
    changed = True
    while changed:
        changed = False
        for p in range(TOTAL_POINTS):
            if (final_mask & (1 << p)) == 0 and (B_mask_final & (1 << p)) == 0:
                # Add p
                final_mask |= (1 << p)
                final_set.append(p)
                # Update B
                for u in final_set[:-1]: # all except p
                    mask_u = iso_mask[p][u]
                    w = 0
                    m = mask_u
                    while m:
                        if m & 1:
                            if block_count_final[w] == 0:
                                B_mask_final |= (1 << w)
                            block_count_final[w] += 1
                        m >>= 1
                        w += 1
                changed = True
                break # Restart loop as B changed

    # Return list of (x,y)
    result = []
    for p in final_set:
        x, y = coords[p]
        result.append((x, y))
        
    return result
