import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    # Helper to check validity of a set of indices
    # Returns True if valid, False otherwise
    def is_valid(selected):
        if len(selected) < 3:
            return True
        selected_arr = np.array(selected)
        # dist2[selected_arr, :] gives distances between all pairs in selected
        # We need to check for isosceles triangles.
        # An isosceles triangle exists if there are 3 points A, B, C such that
        # d(A,B) == d(A,C) or d(A,B) == d(B,C) or d(A,C) == d(B,C).
        
        # Efficient check using sets of distances for each point.
        # For each point i in selected:
        #   Get distances to all other points j in selected.
        #   If any distance is repeated, it's isosceles with apex at i.
        
        # This is O(K^2) per check if done naively with Python loops, 
        # but we can vectorize the distance retrieval.
        
        # Let's use a slightly faster Python-based check for K ~ 20-50
        # which is acceptable for SA moves if we cache or optimize.
        
        # Actually, for SA, we need fast validity checks for single moves.
        # We will maintain the state incrementally.
        pass

    # State representation for SA:
    # selected: list of indices
    # point_dists: list of sets, point_dists[i] contains squared distances from selected[i] to all other selected points.
    
    # To support SA efficiently, we need:
    # 1. Add point p: Check if valid. If yes, update state.
    # 2. Remove point r: Update state.
    # 3. Swap r with p: Remove r, Add p.
    
    # Let's implement a class-like structure or just functions that modify state in place and return delta info?
    # Given the complexity, let's stick to a simpler "check validity of new set" but optimized.
    # However, full re-validation is O(K^2). K~50 -> 2500 ops. 
    # 10,000 moves * 2500 = 25M ops. ~5-10s in Python? Maybe too slow for 300s if we want many moves.
    
    # Better: Incremental validity check.
    
    # Let's try a different approach:
    # 1. Generate a good initial set using a modular lattice.
    # 2. Use a "Greedy Add" phase to fill it up.
    # 3. Use "Swap" moves to improve.
    
    # Modular Lattice Strategy:
    # Points (x, y) such that x + 2y is a multiple of some prime?
    # Or x * a + y * b mod p == 0?
    # Let's try: x = 2*y % 13? No, that's a line.
    # We want a 2D spread.
    # Try: (x, y) where x % 5 == y % 5? No, too dense.
    # Try: x % 7 == 2 * y % 7?
    
    # Let's just use a few deterministic patterns and pick the best one, then refine.
    
    patterns = []
    
    # Pattern 1: Sparse grid
    # (x, y) where x % 4 == 0 and y % 4 == 0? Too sparse.
    
    # Let's use the "Diagonal" idea:
    # (x, y) where x - y is in a specific set of residues mod M?
    
    # Actually, the inspiration mentioned "diagonal or modular lattice".
    # Let's try: x = (2 * y) % 13 + 13 * (y // 13)? No.
    
    # Let's try a simple structured start:
    # Start with empty set, then add points in a specific order that minimizes conflicts.
    # Order: Sort by x, then y.
    # But random restarts already did that with random order.
    
    # Let's try a "Greedy with Lookahead" or "Backtracking" for the initial phase?
    # No, backtracking is expensive.
    
    # Let's try Simulated Annealing directly from a random start, but with better moves.
    # Move types:
    # 1. Add a random point if valid.
    # 2. Remove a random point.
    # 3. Swap a random point with another random point.
    
    # To make SA work, we need fast validity checks.
    # Let's precompute for each point p, the list of points that form an isosceles with p and any other point?
    # No, that's 3-point dependency.
    
    # Let's optimize the validity check for a specific move.
    # When adding p to set S:
    # - Distances from p to all s in S must be unique.
    # - For each s in S, dist(p,s) must not be in dists_from_s.
    
    # We can maintain:
    # S: list of indices
    # dist_sets: list of sets, dist_sets[i] = {dist2[S[i], S[j]] for j != i}
    
    # Add p:
    # 1. Compute d_vals = dist2[p, S]
    # 2. Check if len(set(d_vals)) == len(S). If not, invalid.
    # 3. Check if any d_vals[i] in dist_sets[i]. If so, invalid.
    
    # This is O(K) + O(K log K) for set creation?
    # set(d_vals) is O(K).
    # Checking membership is O(1) avg.
    # So Add check is O(K).
    
    # Remove r (index idx in S):
    # 1. Remove r from S.
    # 2. For each remaining point s in S, remove dist2[r, s] from dist_sets[s_idx].
    # This is O(K).
    
    # So SA moves are O(K).
    # K ~ 50.
    # 100,000 moves * 50 ops = 5M ops. Very fast.
    
    # Let's implement SA.
    
    def get_initial_set():
        # Try a few structured starts
        candidates = []
        
        # Start 1: Random
        random.shuffle(range(TOTAL))
        
        # Start 2: Modular
        # (x, y) such that x % 3 == y % 3?
        mod_points = []
        for x in range(N):
            for y in range(N):
                if x % 3 == y % 3:
                    mod_points.append(y * N + x)
        random.shuffle(mod_points)
        
        # Start 3: Sparse Grid
        sparse_points = []
        for x in range(0, N, 2):
            for y in range(0, N, 2):
                sparse_points.append(y * N + x)
        random.shuffle(sparse_points)
        
        # We will just run greedy on these orders and pick the best result as the initial state for SA.
        
        best_greedy = []
        best_greedy_dists = []
        
        for order in [list(range(TOTAL)), mod_points, sparse_points]:
            random.shuffle(order)
            selected = []
            point_dists = []
            
            for p in order:
                valid = True
                if selected:
                    dvals = dist2[p, selected]
                    if len(set(dvals)) != len(dvals):
                        valid = False
                    else:
                        for i, s in enumerate(selected):
                            if dvals[i] in point_dists[i]:
                                valid = False
                                break
                
                if valid:
                    selected.append(p)
                    if selected:
                        new_dists = set()
                        if len(selected) > 1:
                            new_dists = set(dvals)
                            for i in range(len(selected) - 1):
                                point_dists[i].add(dvals[i])
                        point_dists.append(new_dists)
                    else:
                        point_dists.append(set())
            
            if len(selected) > len(best_greedy):
                best_greedy = selected
                best_greedy_dists = point_dists
                
        return best_greedy, best_greedy_dists

    # Initialize
    selected, point_dists = get_initial_set()
    current_size = len(selected)
    
    if current_size > best_size:
        best_size = current_size
        best_set = [(idx // N, idx % N) for idx in selected]
    
    # Simulated Annealing
    # Temperature T, cooling rate
    T = 1.0
    cooling = 0.999
    
    # We need a way to pick a random move.
    # Moves:
    # 1. Add: Pick random p not in selected. Check validity.
    # 2. Remove: Pick random r in selected.
    # 3. Swap: Pick random r in selected, random p not in selected.
    
    # To make "Add" efficient, we can maintain a list of "candidate" points?
    # Or just pick random index from 0..TOTAL-1 and check if in selected.
    # Checking if in selected: use a set for selected indices.
    
    selected_set = set(selected)
    
    # Mapping from index in selected list to its position?
    # We need to update point_dists when removing/swapping.
    # It's easier to keep selected as a list and point_dists as a list aligned with it.
    # When removing index i from selected:
    #   Remove selected[i] from selected_set.
    #   For all j != i, remove dist2[selected[i], selected[j]] from point_dists[j].
    #   Remove selected[i] from selected list.
    #   Remove point_dists[i] from point_dists list.
    #   Update indices? If we just pop, indices shift.
    #   To avoid shifting, we can use a dict or just accept the O(K) shift.
    #   Since K is small, shifting is fine.
    
    def try_add(p):
        nonlocal selected, point_dists, selected_set
        if p in selected_set:
            return False, 0
        
        dvals = dist2[p, selected]
        if len(set(dvals)) != len(dvals):
            return False, 0
        
        for i, s in enumerate(selected):
            if dvals[i] in point_dists[i]:
                return False, 0
        
        # Valid
        selected.append(p)
        selected_set.add(p)
        
        new_dists = set(dvals)
        for i in range(len(selected) - 1):
            point_dists[i].add(dvals[i])
        
        point_dists.append(new_dists)
        return True, 1

    def try_remove(idx):
        nonlocal selected, point_dists, selected_set
        if idx < 0 or idx >= len(selected):
            return False, 0
        
        p = selected[idx]
        selected_set.remove(p)
        
        # Remove distances from other points
        for i in range(len(selected)):
            if i == idx:
                continue
            d = dist2[p, selected[i]]
            point_dists[i].discard(d)
        
        selected.pop(idx)
        point_dists.pop(idx)
        return True, -1

    def try_swap(idx, p):
        nonlocal selected, point_dists, selected_set
        if p in selected_set:
            return False, 0
        
        r = selected[idx]
        
        # Check if p can be added if r was not there?
        # We need to check validity of (selected - {r}) + {p}
        
        # Temporarily remove r
        # This is complex to do without actually removing.
        # Let's just do:
        # 1. Remove r
        # 2. Try add p
        # 3. If add fails, add r back
        
        # Remove r
        selected_set.remove(r)
        
        # Update point_dists for removal of r
        for i in range(len(selected)):
            if i == idx:
                continue
            d = dist2[r, selected[i]]
            point_dists[i].discard(d)
        
        # Save state to rollback if needed?
        # Or just check validity of p against (selected - {r})
        
        # Let's create a helper to check validity of p against a list of indices and their dist sets
        # But point_dists is modified in place.
        
        # Alternative:
        # Compute dvals_p = dist2[p, selected_without_r]
        # Check uniqueness
        # Check conflicts with point_dists of selected_without_r (which are currently updated to exclude r)
        
        # Let's do it carefully.
        
        # We already removed r from selected_set and updated point_dists to exclude r.
        # Now selected list still has r at idx.
        # Let's pop r from list.
        selected.pop(idx)
        point_dists.pop(idx)
        
        # Now try to add p
        success, delta = try_add(p)
        
        if not success:
            # Rollback: Add r back
            # We need to restore r to its original position?
            # Or just add it to the end?
            # If we add to end, the indices change.
            # It's safer to just revert the removal and the add attempt.
            
            # Revert Add attempt (if it partially succeeded? No, try_add is atomic)
            # So we just need to revert the Removal.
            
            # Re-add r to selected and point_dists
            selected.append(r) # Added to end, not original idx
            selected_set.add(r)
            
            # Update point_dists
            new_dists = set()
            for i, s in enumerate(selected[:-1]):
                d = dist2[r, s]
                new_dists.add(d)
                point_dists[i].add(d)
            
            point_dists.append(new_dists)
            
            # Note: The order of selected is now different (r is at end, not idx).
            # This is fine for the set, but our "idx" based moves might be confused if we track indices.
            # However, we don't track indices across moves, we pick random indices in the current list.
            # So this is acceptable.
            
            return False, 0
        
        return True, 0 # Size unchanged

    # SA Loop
    num_moves = 50000
    move_count = 0
    
    while move_count < num_moves:
        if time.time() > deadline:
            break
            
        if T < 0.01:
            # Only accept improvements
            pass
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            p = random.randint(0, TOTAL - 1)
            success, delta = try_add(p)
            if success:
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = [(idx // N, idx % N) for idx in selected]
        elif move_type < 0.7:
            # Remove
            if selected:
                idx = random.randint(0, len(selected) - 1)
                success, delta = try_remove(idx)
                if success:
                    current_size += delta
        else:
            # Swap
            if selected:
                idx = random.randint(0, len(selected) - 1)
                p = random.randint(0, TOTAL - 1)
                success, delta = try_swap(idx, p)
                if success:
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = [(idx // N, idx % N) for idx in selected]
        
        # Cooling
        T *= cooling
        move_count += 1
        
        # Periodic restart if stuck?
        # If no improvement in last 1000 moves, restart SA with a new initial set?
        # Let's keep it simple.
        
    return best_set

# Run construct to verify it works
# Note: The evaluator calls construct()
