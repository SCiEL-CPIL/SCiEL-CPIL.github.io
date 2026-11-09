N = 16
NUM_PTS = N * N

# Precompute geometry
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Map point index to (x,y)
# p_idx = y * N + x

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute blocked masks for each pair
# blocked_mask[i][j] is a bitmask of points k such that triangle i,j,k is isosceles
# Note: if k == i or k == j, distance is 0. 
# If k is distinct from i,j:
# d(i,j) == d(i,k) OR d(i,j) == d(j,k) OR d(i,k) == d(j,k)

blocked_masks = [[0] * NUM_PTS for _ in range(NUM_PTS)]

for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i][j]
        mask = 0
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        blocked_masks[i][j] = mask
        blocked_masks[j][i] = mask

def is_valid(set_list):
    """Check if no three points in set_list form an isosceles triangle."""
    n = len(set_list)
    for i in range(n):
        pi = set_list[i]
        for j in range(i + 1, n):
            pj = set_list[j]
            mask_ij = blocked_masks[pi][pj]
            for k in range(j + 1, n):
                if (mask_ij >> set_list[k]) & 1:
                    return False
    return True

def get_blocked_mask(set_list):
    """Return the union of blocked masks for all pairs in set_list."""
    mask = 0
    n = len(set_list)
    for i in range(n):
        pi = set_list[i]
        for j in range(i + 1, n):
            pj = set_list[j]
            mask |= blocked_masks[pi][pj]
    return mask

def is_maximal(set_list):
    """Check if no point outside set_list can be added."""
    if not is_valid(set_list):
        return False
    
    set_mask = 0
    for p in set_list:
        set_mask |= (1 << p)
    
    blocked = get_blocked_mask(set_list)
    
    coverage = set_mask | blocked
    full_mask = (1 << NUM_PTS) - 1
    return (coverage & full_mask) == full_mask

def construct():
    import time
    import random
    import itertools
    
    DEADLINE = time.time() + 295
    
    # 1. Generate initial valid sets
    best_set = None
    best_len = float('inf')
    
    # Greedy construction
    def greedy_construct(seed):
        rng = random.Random(seed)
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        
        S = []
        S_mask = 0
        
        for p in order:
            # Check if p can be added to S
            # It must not be blocked by any pair in S?
            # No, it must not form an isosceles triangle with any pair in S.
            # i.e., for all pairs (a,b) in S, p must NOT be in blocked_masks[a][b].
            
            can_add = True
            # Optimization: maintain current blocked mask?
            # But adding p changes the blocked mask for future points.
            # For now, just check against existing pairs.
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if (blocked_masks[S[i]][S[j]] >> p) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.append(p)
                S_mask |= (1 << p)
                
        return S

    # Run multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
        S = greedy_construct(i)
        # Check if maximal
        if is_maximal(S):
            if len(S) < best_len:
                best_len = len(S)
                best_set = S.copy()
        else:
            # Try to make it maximal by adding points
            # Greedy fill
            S_mask = 0
            for p in S:
                S_mask |= (1 << p)
            
            # Find unblocked points
            blocked = get_blocked_mask(S)
            unblocked_candidates = []
            for p in range(NUM_PTS):
                if not (S_mask >> p) & 1:
                    if not (blocked >> p) & 1:
                        unblocked_candidates.append(p)
            
            # Add them greedily? This will make it maximal but likely larger.
            # We want minimal maximal.
            # If we add points, size increases.
            # Maybe we should just keep it if it's small, but it's not maximal.
            # The task REQUIRES maximal.
            # So we MUST add points to make it maximal.
            # To minimize size, we should pick points that block the most other unblocked points.
            
            if unblocked_candidates:
                # Sort by how many unblocked points they block?
                # This is expensive. Just add randomly or first found.
                # For now, add all? No, that makes it huge.
                # Let's just add one that blocks the most, repeatedly.
                
                while unblocked_candidates and time.time() < DEADLINE:
                    # Find point in unblocked_candidates that blocks most other unblocked_candidates
                    # Or just pick one
                    p_add = unblocked_candidates[0]
                    
                    # Check validity
                    can_add = True
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            if (blocked_masks[S[i]][S[j]] >> p_add) & 1:
                                can_add = False
                                break
                        if not can_add:
                            break
                    
                    if can_add:
                        S.append(p_add)
                        S_mask |= (1 << p_add)
                        unblocked_candidates.remove(p_add)
                        
                        # Update unblocked_candidates
                        # Recalculate blocked mask? Expensive.
                        # Just recalculate is_maximal later?
                        # For this quick fill, let's just mark it.
                        # Actually, adding p_add blocks MORE points.
                        # So some points in unblocked_candidates might now be blocked.
                        # We should re-evaluate.
                        pass
                    else:
                        unblocked_candidates.remove(p_add)
                        
                # After filling, check if maximal
                if is_maximal(S):
                    if len(S) < best_len:
                        best_len = len(S)
                        best_set = S.copy()

    # 2. Local Search: Remove points
    if best_set is not None:
        S = best_set
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            n = len(S)
            # Try removing each point
            for i in range(n):
                S_trial = S[:i] + S[i+1:]
                if is_maximal(S_trial):
                    S = S_trial
                    improved = True
                    break
    
    # 3. Local Search: Swaps
    if best_set is not None:
        S = best_set
        max_iters = 100
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            n = len(S)
            S_mask = 0
            for p in S:
                S_mask |= (1 << p)
            
            for i in range(n):
                x = S[i]
                S_without_x = S[:i] + S[i+1:]
                
                # Try adding some y not in S
                # Randomly pick a few y
                candidates = [y for y in range(NUM_PTS) if not (S_mask >> y) & 1]
                random.shuffle(candidates)
                
                for y in candidates[:5]: # Try first 5
                    S_swap = S_without_x + [y]
                    # Check validity
                    if is_valid(S_swap):
                        # Check if this swap enables a removal?
                        # Or just accept if it's valid.
                        # We want to eventually remove points.
                        # So just accept valid swaps to explore landscape.
                        S = S_swap
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
            
            # After swaps, try to remove points again
            improved2 = True
            while improved2 and time.time() < DEADLINE:
                improved2 = False
                n = len(S)
                for i in range(n):
                    S_trial = S[:i] + S[i+1:]
                    if is_maximal(S_trial):
                        S = S_trial
                        improved2 = True
                        break
            
            if len(S) < best_len:
                best_len = len(S)
                best_set = S.copy()

    if best_set is None:
        # Fallback
        best_set = greedy_construct(0)
        # Ensure maximal
        # (The fallback might not be maximal, but we need to return something)
        # For safety, just return it. The evaluator might penalize non-maximal? 
        # "forming a maximal no-isosceles set". 
        # If not maximal, it's invalid. 
        # So we MUST ensure maximality.
        
        # Quick fix: add points until maximal
        S_mask = 0
        for p in best_set:
            S_mask |= (1 << p)
        
        blocked = get_blocked_mask(best_set)
        while True:
            # Find unblocked point
            unblocked = None
            for p in range(NUM_PTS):
                if not (S_mask >> p) & 1 and not (blocked >> p) & 1:
                    unblocked = p
                    break
            
            if unblocked is None:
                break
            
            # Check if unblocked can be added (validity)
            can_add = True
            for i in range(len(best_set)):
                for j in range(i + 1, len(best_set)):
                    if (blocked_masks[best_set[i]][best_set[j]] >> unblocked) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                best_set.append(unblocked)
                S_mask |= (1 << unblocked)
                blocked |= get_blocked_mask([unblocked]) | 0 # This is wrong, need to update blocked correctly
                # Recalculate blocked
                blocked = get_blocked_mask(best_set)
            else:
                # Skip this point? If we can't add it, it's effectively "blocked" by validity constraint?
                # No, if it's unblocked by geometry but invalid by set, it CANNOT be added.
                # So it does NOT violate maximality?
                # Maximal means "no FURTHER grid point can be ADDED".
                # If it cannot be added due to isosceles constraint, it's effectively blocked.
                # So if `can_add` is False, it's fine. It doesn't need to be in `blocked` mask.
                # The `blocked` mask represents "cannot be added because of isosceles with existing pairs".
                # If it's not in `blocked`, it means it CAN be added geometrically.
                # But if it's invalid, it means it CANNOT be added.
                # So the condition for maximality is:
                # For all p not in S: (p in blocked) OR (p is invalid with S).
                
                # My `is_maximal` check:
                # `is_valid(S)` is true.
                # `coverage = set_mask | blocked`.
                # If `coverage` is full, it's maximal.
                # Wait. If `p` is not in `blocked`, but `p` is invalid with `S`, then `p` is NOT addable.
                # But my `is_maximal` only checks `blocked`.
                # If `p` is not in `blocked`, `is_maximal` says "p can be added" (geometrically).
                # But if `p` is invalid, it CANNOT be added.
                # So my `is_maximal` is WRONG.
                
                # Correct Maximality:
                # A set S is maximal if for all p not in S, p cannot be added to S.
                # p cannot be added if:
                # 1. p forms isosceles with some pair in S (p in blocked_mask[S])
                # OR
                # 2. p is the same as some point in S (p in S)
                
                # Wait. If p is not in S, and p is NOT in blocked_mask[S], does that mean p CAN be added?
                # Yes, adding p to S creates a new set S' = S U {p}.
                # S' is valid if no isosceles triangles in S'.
                # Since S is valid, we only need to check triangles involving p.
                # These are exactly the conditions checked by blocked_mask[S].
                # If p is not in blocked_mask[S], then NO pair in S forms an isosceles with p.
                # Therefore, S' IS VALID.
                # Therefore, p CAN be added.
                # So if p is not in blocked_mask[S] and p not in S, S is NOT maximal.
                
                # So my `is_maximal` logic `coverage = set_mask | blocked` IS correct.
                
                # Why did I think it was wrong?
                # "If p is invalid with S" -> This implies p IS in blocked_mask[S].
                # Because "invalid" means it forms an isosceles triangle with some pair in S.
                # So `can_add` being False implies `p in blocked_mask[S]`.
                # So if `unblocked` is found (not in blocked), `can_add` MUST be True.
                # So the `else` branch is never reached if `blocked` is calculated correctly.
                
                pass

    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
