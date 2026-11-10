import time
import random
import math

N = 64
TIME_LIMIT = 290 # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    # Precompute squared distances between all pairs? 
    # 64*64 = 4096 points. 4096^2 / 2 ~ 8 million pairs.
    # Memory: 8 million integers. ~64MB. Acceptable?
    # Speed: Precomputing takes time. 
    # On-the-fly calculation is fast enough for k~10.
    # Let's stick to on-the-fly to avoid precomputation overhead and memory.
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def rebuild_dist_from(selected):
        """
        selected: list of points
        returns: list of sets, where dist_from[i] contains dist2(selected[i], selected[j]) for all j != i
        """
        dist_from = []
        for i in range(len(selected)):
            s = set()
            pi = selected[i]
            for j in range(len(selected)):
                if i == j: continue
                pj = selected[j]
                s.add(dist2(pi, pj))
            dist_from.append(s)
        return dist_from

    def try_add(p, selected, dist_from):
        """
        Returns True if p can be added, and updates selected/dist_from in place if so.
        """
        # 1. Check p is not equidistant to any two existing points
        d_p = []
        for pt in selected:
            d_p.append(dist2(p, pt))
            
        if len(set(d_p)) != len(d_p):
            return False
            
        # 2. Check for each existing point u, d(p,u) is not equal to any d(u,v)
        for i in range(len(selected)):
            if d_p[i] in dist_from[i]:
                return False
                
        # Valid. Add it.
        selected.append(p)
        # Create dist set for new point
        new_dist_set = set(d_p)
        dist_from.append(new_dist_set)
        # Update existing dist sets
        for i in range(len(selected) - 1):
            dist_from[i].add(d_p[i])
            
        return True

    def try_remove(idx, selected, dist_from):
        """
        Removes point at idx from selected and dist_from.
        """
        p = selected[idx]
        selected.pop(idx)
        dist_from.pop(idx)
        
        # We need to remove dist(p, u) from dist_from[u] for all remaining u.
        # Note: indices in dist_from have shifted for elements after idx.
        # It's safer to rebuild if we remove, or carefully update.
        # Given k is small, rebuilding is O(k^2) which is fine.
        # Or we can just update.
        # Let's rebuild for simplicity and correctness, as removals are less frequent than checks in the inner loop?
        # Actually, in the swap logic, we remove one and add one.
        # If we rebuild after remove, then check add, then we are safe.
        # Rebuild is O(k^2). For k=10, 100 ops. Very fast.
        new_dist_from = rebuild_dist_from(selected)
        # Replace contents
        dist_from.clear()
        dist_from.extend(new_dist_from)

    def greedy_fill(selected, dist_from, candidates):
        """
        Greedily add points from candidates to selected/dist_from.
        candidates: list of points not in selected.
        """
        # We iterate through candidates. If one fits, add it and remove from candidates.
        # To be effective, we should probably shuffle candidates or pick random ones.
        # However, standard greedy is deterministic. 
        # Let's use a randomized greedy: pick a random candidate, try to add.
        # If it fails, move to next.
        
        # To make it fast, we can just iterate through a shuffled list of candidates.
        # But candidates changes as we add points.
        
        # Better approach for local search:
        # We have a maximal set. We try to swap.
        
        pass # Not used directly as a separate function in the main loop below, 
             # but the logic is embedded.

    def local_search(initial_set, initial_dist):
        """
        Perform local search starting from initial_set.
        Returns the best set found in this trajectory.
        """
        selected = list(initial_set)
        dist_from = [set(s) for s in initial_dist] # Deep copy
        
        current_size = len(selected)
        local_best = list(selected)
        local_best_size = current_size
        
        # Build set of points for O(1) lookup
        selected_set = set(selected)
        
        # Precompute list of all points not in selected
        # This is expensive if done every iteration.
        # We can maintain a list of 'available' points.
        
        # Strategy:
        # 1. Try to add any point (should fail if maximal, but maybe we started non-maximal).
        # 2. Try swaps: remove u, add v.
        
        # To speed up swap search:
        # For each u in selected:
        #   Remove u
        #   For each v in all_points not in selected:
        #      Try to add v
        #      If success:
        #          Accept swap (or with prob)
        #          Update best if size increased (it won't, size stays same)
        #          Then try to add more points (greedy fill)
        #          If size increased, update best
        #          Break out of v loop? Or continue?
        #   Re-add u (or rebuild)
        
        # This is O(k * N^2 * k) per step? No.
        # N^2 = 4096. k ~ 10.
        # 10 * 4096 * 10 = 409,600 ops per full sweep of swaps.
        # This is fast enough.
        
        max_swaps = 5000 # Limit swaps per trajectory to avoid infinite loops
        
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
                
            # Try to add any point first (in case we are not maximal)
            # This is O(N^2 * k). 
            # We can skip this if we know we are maximal.
            # Let's assume we start maximal from greedy.
            
            # Perform a random swap attempt
            # Pick a random point to remove
            idx_u = rng.randint(0, len(selected) - 1)
            u = selected[idx_u]
            
            # Remove u
            try_remove(idx_u, selected, dist_from)
            
            # Try to add a random point v
            # We need a point not in selected.
            # Picking a random point from all_points and checking if it's in selected is O(1) average.
            # But we need to try to ADD it.
            
            # To increase chance of success, we can try multiple v's?
            # Or just one random v.
            # If we just pick one random v, we might fail often.
            # Let's pick a random v from all_points.
            
            v = rng.choice(all_points)
            
            if v in selected_set:
                # We removed u, so v cannot be in selected unless v==u (impossible since we removed u)
                # Wait, selected_set needs to be updated.
                pass
            
            # Update selected_set
            selected_set.discard(u)
            
            if v not in selected_set:
                if try_add(v, selected, dist_from):
                    # Swap successful
                    selected_set.add(v)
                    # Try to fill further
                    # We can do a quick greedy fill here
                    # But to keep it simple and fast, just accept the swap.
                    # The next iteration will try another swap.
                    # However, if we can add more points, we should.
                    # Let's do a limited greedy fill: try to add up to 5 random points.
                    for _ in range(5):
                        w = rng.choice(all_points)
                        if w not in selected_set:
                            if try_add(w, selected, dist_from):
                                selected_set.add(w)
                                if len(selected) > local_best_size:
                                    local_best_size = len(selected)
                                    local_best = list(selected)
                            else:
                                # Failed to add w, try another?
                                pass
                    
                    # Check if we found a better size
                    if len(selected) > local_best_size:
                        local_best_size = len(selected)
                        local_best = list(selected)
                        
                    # If size didn't increase, we are just moving around.
                    # This is a random walk.
                    
                    # If we have time, we can continue.
                    # To avoid getting stuck in a cycle, we could track visited states, 
                    # but that's complex. Random restarts help.
                    
                    # If we improved global best, we can break or continue.
                    # Let's continue to see if we can improve further.
                    
                    # Note: try_add modifies selected and dist_from.
                    # We must ensure dist_from is consistent.
                    # try_add updates it correctly.
                    
                else:
                    # Swap failed. Re-add u.
                    # We need to restore u to selected and dist_from.
                    # try_add(u, selected, dist_from) should work if valid.
                    # Since (selected without u) + u was valid originally, it must be valid now.
                    if not try_add(u, selected, dist_from):
                        # This should not happen
                        pass
                    selected_set.add(u)
            else:
                # v was in selected? 
                # We removed u. If v == u, we are re-adding u.
                # If v != u, v is in selected. We can't add it.
                # Re-add u.
                if not try_add(u, selected, dist_from):
                    pass
                selected_set.add(u)
            
            # If we never successfully swapped, we might loop.
            # The logic above handles success and failure.
            
        return local_best, local_best_size

    # Main Loop: Multiple Restarts
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # 1. Generate initial set via Randomized Greedy
        selected = []
        dist_from = []
        candidates = list(range(len(all_points)))
        rng.shuffle(candidates)
        
        for idx in candidates:
            p = all_points[idx]
            if try_add(p, selected, dist_from):
                pass # Added
            if time.time() > deadline - 5:
                break
        
        # Now selected is a maximal set (or close to it).
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(selected)
        
        # 2. Local Search
        # Only do local search if we have time and if the current set is promising
        # (e.g., size is close to best, or just always do it)
        
        if time.time() < deadline - 20:
            # Rebuild dist_from to be safe (greedy try_add maintains it, but let's ensure)
            dist_from = rebuild_dist_from(selected)
            
            local_best, local_best_size = local_search(selected, dist_from)
            
            if local_best_size > best_size:
                best_size = local_best_size
                best_set = list(local_best)
        
        # 3. If best_size is high enough, we can stop or continue?
        # The problem is likely to have a small maximum (e.g., 10-20).
        # If we find 15, we might keep searching for 16.
        # Given the time limit, we should keep searching until deadline.

    return best_set

# Test/Verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
