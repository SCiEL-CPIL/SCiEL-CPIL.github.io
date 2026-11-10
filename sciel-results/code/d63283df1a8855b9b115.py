"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Precompute coordinates and squared distances
num_points = N * N
coords = np.array([(i // N, i % N) for i in range(num_points)])

# dist[i][j] is squared distance between point i and point j
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sum(diff**2, axis=2).astype(np.int32)

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle (including degenerate)."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def can_add(p, S_set):
    """Check if point p can be added to set S_set without creating an isosceles triangle."""
    if len(S_set) < 2:
        return True
        
    s_list = list(S_set)
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i + 1, len(s_list)):
            b = s_list[j]
            if is_isosceles(p, a, b):
                return False
    return True

def is_maximal(S_set):
    """Check if no point can be added to S_set."""
    for p in range(num_points):
        if p not in S_set:
            if can_add(p, S_set):
                return False
    return True

def make_maximal(S_set):
    """Add points to S_set until it is maximal. Returns the new set."""
    # We need to be careful with order. 
    # A standard greedy maximal extension:
    # While there exists a point that can be added, add it.
    # Since adding a point can only block others, a single pass in fixed order 
    # followed by checking if any remain addable is not sufficient if the order 
    # causes a point to be skipped that would have been valid if added earlier.
    # However, if we just iterate through all points and add if valid, 
    # is it maximal? 
    # Let S be the set after one pass. 
    # If p was not added, it was either in S (impossible if we only add) or 
    # blocked by some pair in S_at_that_time.
    # S_at_that_time is a subset of S.
    # So p is blocked by a pair in S.
    # Thus p cannot be added to S.
    # So one pass is sufficient to reach a maximal set?
    # Yes. If p is not added, it is invalid w.r.t S_subset. 
    # Since S_subset subset S, if p is invalid w.r.t S_subset, it is invalid w.r.t S.
    # So the set is maximal.
    
    new_set = set(S_set)
    for p in range(num_points):
        if p not in new_set:
            if can_add(p, new_set):
                new_set.add(p)
    return new_set

def construct():
    start_time = time.time()
    time_limit = 295  # 5s margin
    
    best_size = num_points
    best_set = None
    
    random.seed(42)
    
    # Simulated Annealing parameters
    T = 100.0
    T_min = 0.1
    alpha = 0.999
    
    # Generate initial random maximal set
    order = list(range(num_points))
    random.shuffle(order)
    current_set = set()
    for p in order:
        if can_add(p, current_set):
            current_set.add(p)
    current_set = make_maximal(current_set)
    
    if len(current_set) < best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    step = 0
    while time.time() - start_time < time_limit:
        step += 1
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Try a 2-swap move: Remove 2 points, Add 2 points
        # To ensure the set remains a valid independent set during the move,
        # we must ensure the new points don't form isosceles with the remaining old points
        # and with each other.
        
        # 1. Select 2 points to remove
        # Random selection from current set
        if len(current_set) < 4:
            # Not enough to swap, try simple removal?
            # For minimal sets, size is likely > 4.
            pass
            
        removal_candidates = list(current_set)
        random.shuffle(removal_candidates)
        if len(removal_candidates) < 2:
            break
        p1, p2 = removal_candidates[0], removal_candidates[1]
        
        # 2. Identify points that can be added to (current_set - {p1, p2})
        # Let S' = current_set - {p1, p2}
        S_prime = current_set - {p1, p2}
        
        # Find all q not in S' such that can_add(q, S_prime)
        # We want to pick 2 such q's that are also compatible with each other.
        
        # To be efficient, we can just iterate through candidates.
        # Given N=16, num_points=256.
        # Checking can_add is O(|S'|^2) ~ O(256^2) worst case? No, |S'| is small (maybe 10-20).
        # So can_add is fast.
        
        valid_additions = []
        for q in range(num_points):
            if q in S_prime:
                continue
            if can_add(q, S_prime):
                valid_additions.append(q)
                
        if len(valid_additions) < 2:
            continue
            
        # Pick 2 random from valid_additions
        # Shuffle to pick 2 distinct
        random.shuffle(valid_additions)
        # Ensure we pick 2 distinct
        q1, q2 = valid_additions[0], valid_additions[1]
        
        # Check if q1 and q2 are compatible with each other and with S_prime
        # We already know can_add(q1, S_prime) and can_add(q2, S_prime).
        # We need to check if {q1, q2, s} is isosceles for any s in S_prime?
        # No, can_add(q1, S_prime) ensures q1 is compatible with all pairs in S_prime.
        # Same for q2.
        # We need to check if q1, q2 form an isosceles triangle with any point in S_prime?
        # Wait. The condition for a set to be independent is that NO triple in the set is isosceles.
        # If we add q1 and q2 to S_prime, we introduce new triples:
        # 1. (q1, q2, s) for s in S_prime.
        # 2. (q1, q2, q1) - degenerate, usually ignored or d=0. 
        #    is_isosceles(q1, q2, q1) checks dist(q1,q2), dist(q1,q1)=0, dist(q2,q1).
        #    d1=d2? dist(q1,q2)==0? No.
        #    d1=d3? dist(q1,q2)==dist(q2,q1)? Yes.
        #    So (q1, q2, q1) is considered isosceles by the function?
        #    Let's check is_isosceles(i, j, k).
        #    d1 = dist[i][j], d2 = dist[i][k], d3 = dist[j][k].
        #    If i=k, d2=0, d3=dist[j][i]=d1. So d1==d3 is True.
        #    So the function returns True for (i, j, i).
        #    This means we cannot have duplicate points, which is true for a set.
        #    But does it prevent adding q1 and q2 if they are distinct?
        #    The triple (q1, q2, s) where s is in S_prime.
        #    can_add(q1, S_prime) checks all pairs (a,b) in S_prime.
        #    It does NOT check the pair (q2, s) because q2 is not in S_prime yet.
        #    So we MUST explicitly check compatibility of q1 and q2 with each other 
        #    in the context of S_prime.
        
        # Check if adding {q1, q2} to S_prime creates any isosceles triangle.
        # The new triples involving q1 or q2 are:
        # - (q1, q2, s) for s in S_prime
        # - (q1, s1, s2) for s1, s2 in S_prime -> Checked by can_add(q1, S_prime)
        # - (q2, s1, s2) for s1, s2 in S_prime -> Checked by can_add(q2, S_prime)
        # - (q1, q2, q1) -> Degenerate, irrelevant for set logic usually, but function says isosceles.
        #   However, we are adding distinct points q1, q2.
        #   The set will contain q1, q2, and S_prime.
        #   We need to ensure no triple in {q1, q2} U S_prime is isosceles.
        
        valid = True
        s_list = list(S_prime)
        for s in s_list:
            if is_isosceles(q1, q2, s):
                valid = False
                break
        
        if not valid:
            # Try next pair? For simplicity, just reject this move.
            # To be more robust, we could loop through valid_additions to find a compatible pair.
            continue
            
        # If valid, calculate cost change
        # Old size: |current_set|
        # New size: |current_set| - 2 + 2 = |current_set|
        # Wait, a swap keeps the size the same.
        # We want to MINIMIZE size.
        # A 2-swap that keeps size the same is only useful if it allows for a subsequent 
        # size-reducing move (e.g., a removal that becomes possible).
        # Or, we can do a "1-swap" where we remove 2 and add 1? 
        # Or remove 1 and add 2? (Increases size, bad).
        # Or remove 2 and add 0? (Decreases size by 2).
        # The most powerful move for minimization is **Remove 2, Add 0** (if valid) 
        # or **Remove 2, Add 1** (if valid, net -1).
        
        # Let's try **Remove 2, Add 0** first.
        # This is just checking if S_prime is an independent set.
        # Since current_set is independent, and we removed points, S_prime is independent.
        # So S_prime is always a valid independent set.
        # But is it MAXIMAL?
        # If S_prime is maximal, we found a smaller maximal set!
        # If S_prime is NOT maximal, we can extend it.
        # But extending it will likely bring it back to a similar size or larger.
        # However, if we can extend it to a maximal set of size < current_size, we win.
        
        # Let's try to extend S_prime to a maximal set S_max.
        # If |S_max| < |current_set|, accept.
        
        S_max = make_maximal(S_prime)
        
        if len(S_max) < len(current_set):
            # Improvement!
            current_set = S_max
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            # Reset temperature? Or just continue.
            # Resetting T can help explore new areas.
            T = max(T, 10.0)
            continue
            
        # If no immediate improvement, try a 2-swap (Remove 2, Add 2) to change structure 
        # without changing size, hoping to enable future removals.
        # This is the "Simulated Annealing" part: accept neutral or slightly worse moves 
        # with probability exp(-delta/T).
        # Here delta = 0 for 2-swap. So we always accept 2-swaps?
        # That might lead to random walking.
        # Let's restrict 2-swaps to only those that change the "structure" significantly 
        # or just accept them to explore the space of maximal sets of the same size.
        
        # Actually, a simpler and more effective move for minimization is:
        # **Remove 1 point**.
        # If the resulting set S' can be extended to a maximal set S_max with |S_max| < |current_set|, accept.
        # This is what the previous program did, but it was greedy.
        # Here we can be more systematic or use SA.
        
        # Let's stick to the 2-removal + refill strategy as it's more aggressive.
        # If S_max size == current_size, we might accept the move with some probability 
        # to escape local minima?
        # But S_max is just one possible extension. There might be another extension with smaller size.
        # make_maximal is greedy. 
        # To be more thorough, we could try multiple random extensions of S_prime.
        
        # Given time constraints, let's try a few random extensions of S_prime.
        found_improvement = False
        for _ in range(5):
            # Random greedy extension
            S_ext = set(S_prime)
            remaining = [p for p in range(num_points) if p not in S_ext]
            random.shuffle(remaining)
            for p in remaining:
                if can_add(p, S_ext):
                    S_ext.add(p)
            # S_ext is maximal (by the single-pass argument)
            if len(S_ext) < len(current_set):
                current_set = S_ext
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                found_improvement = True
                break
                
        if found_improvement:
            T = max(T, 10.0)
            continue
            
        # If no improvement found in this step, we could try a 2-swap to perturb.
        # But for now, let's just try another random removal pair in the next iteration.
        # To avoid getting stuck, we can occasionally accept a "neutral" move 
        # (same size) if it changes the set.
        
        # Let's implement a simple 2-swap acceptance.
        # We already have q1, q2 and S_prime.
        # We checked compatibility.
        # New set: S_new = S_prime U {q1, q2}
        # Size is same.
        # Accept with probability 0.5 to explore?
        if random.random() < 0.1:
             current_set = S_prime | {q1, q2}
             # Ensure it's maximal? 
             # It might not be maximal.
             # If we accept a non-maximal set, we must extend it.
             # But extending it might increase size.
             # So we should only accept if it's maximal or if we extend it and size is same/smaller.
             S_ext = make_maximal(current_set)
             if len(S_ext) <= len(current_set): # Should be equal or larger
                 if len(S_ext) < best_size:
                     best_size = len(S_ext)
                     best_set = set(S_ext)
                 current_set = S_ext

    if best_set is None:
        return []
    
    return [tuple(c) for c in coords[list(best_set)]]
# EVOLVE-BLOCK-END
