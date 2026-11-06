"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

all_points = list(itertools.product(range(N), repeat=2))
num_points = len(all_points)

# Precompute iso_third[i][j] = set of k such that (i, j, k) is isosceles
# We need this to check maximality and update block counts.
# Note: (i, j, k) is isosceles if dist(i,j)==dist(i,k) or dist(i,j)==dist(j,k) or dist(j,k)==dist(i,k)
# We store k in iso_third[i][j].
# Symmetry: if k in iso_third[i][j], then j in iso_third[i][k] and i in iso_third[j][k].
# We will fill this array.

iso_third = [[set() for _ in range(num_points)] for _ in range(num_points)]

# Precomputation
# To speed up, we can group points by distance squared?
# Or just brute force 256^3?
# 256^3 = 16M. In Python, nested loops are slow.
# Let's optimize by iterating i, j and calculating d_ij.
# Then for each k, check conditions.
# We can precompute distances between all pairs? 256*256/2 = 32k pairs.
# dist_sq[i][j]
dist_sq = [[0]*num_points for _ in range(num_points)]
for i in range(num_points):
    xi, yi = all_points[i]
    for j in range(i+1, num_points):
        xj, yj = all_points[j]
        d = (xi-xj)**2 + (yi-yj)**2
        dist_sq[i][j] = d
        dist_sq[j][i] = d

# Fill iso_third
for i in range(num_points):
    for j in range(num_points):
        if i == j:
            continue
        d_ij = dist_sq[i][j]
        for k in range(num_points):
            if k == i or k == j:
                continue
            d_ik = dist_sq[i][k]
            d_jk = dist_sq[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                iso_third[i][j].add(k)

def construct():
    best_size = float('inf')
    best_set = None
    
    # Time limit check
    start_time = time.time()
    time_limit = 290 # seconds
    
    # Strategy:
    # 1. Generate a maximal no-isosceles set using a randomized greedy.
    # 2. Try to remove points from this set to make it smaller while keeping it maximal.
    
    def generate_initial_maximal():
        chosen = set()
        available = set(range(num_points))
        
        # Randomize order to get different initial sets
        points_list = list(available)
        random.shuffle(points_list)
        
        for p in points_list:
            if p not in available:
                continue
            # Check if adding p creates an isosceles triangle with any pair in chosen
            valid = True
            for a in chosen:
                # If p, a, b is isosceles for any b in chosen, invalid
                # b must be in chosen and b != a
                # We check if any b in chosen is in iso_third[p][a]
                # Actually, if b in chosen and b in iso_third[p][a], then (p,a,b) is isosceles.
                # We just need to check intersection of chosen and iso_third[p][a] (excluding a? No, a!=p)
                # If there is any b in chosen such that b in iso_third[p][a], it's invalid.
                # Note: a is in chosen. Is a in iso_third[p][a]?
                # dist(p,a) == dist(p,a) is true. So (p,a,a) is degenerate?
                # The problem says "grid points". Usually distinct points are implied for triangles.
                # "no-isosceles set" usually means no 3 distinct points form an isosceles triangle.
                # My precomputation skipped k==i or k==j.
                # So iso_third[p][a] does NOT contain p or a.
                # So we just check if chosen intersects iso_third[p][a].
                if chosen & iso_third[p][a]:
                    valid = False
                    break
            if valid:
                chosen.add(p)
                # Remove p from available?
                # Also, any point q that forms isosceles with p and some a in chosen
                # is no longer available?
                # No, available means "can be added".
                # If adding q would create isosceles with current chosen, q is not available.
                # We don't need to explicitly update available if we check validity on the fly
                # against the current `chosen`.
                # However, to speed up, we can maintain `available` as points that *could* be added.
                # But checking validity is O(|chosen|).
                # Let's just iterate through the shuffled list and add if valid.
                
        # The resulting set is maximal?
        # Yes, because we considered every point in the grid (via points_list).
        # If a point wasn't added, it's because it was invalid (created isosceles).
        # So no point can be added.
        return chosen

    def is_maximal(chosen):
        # Check if every point not in chosen is blocked by chosen
        # A point q is blocked if there exist a, b in chosen such that (a, b, q) is isosceles.
        # We can use the block_count method.
        # But for a single check, we can just iterate.
        for q in range(num_points):
            if q in chosen:
                continue
            # Check if q is blocked
            blocked = False
            for a in chosen:
                for b in chosen:
                    if b != a and q in iso_third[a][b]:
                        blocked = True
                        break
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    def try_remove(chosen, block_count):
        # Returns a new set and block_count if a point was removed, else None
        # We try to remove a point p from chosen.
        # Condition 1: p must be blocked by chosen \ {p}
        # Condition 2: All q not in chosen \ {p} must be blocked by chosen \ {p}
        
        # To make this efficient, we maintain block_count for all 256 points.
        # block_count[q] = number of pairs (a,b) in chosen such that (a,b,q) is isosceles.
        
        # We will iterate through points in chosen to find one that can be removed.
        # We pick the one that has the highest "redundancy"?
        # Or just try all?
        # Let's try all points in chosen in random order.
        
        chosen_list = list(chosen)
        random.shuffle(chosen_list)
        
        for p in chosen_list:
            # Check if p is blocked by chosen \ {p}
            # p is blocked if there exist a, b in chosen \ {p} such that (a, b, p) is isosceles.
            # This is equivalent to block_count[p] > 0?
            # Wait, block_count[p] counts pairs in `chosen` that block p.
            # The pairs are (a,b) where a,b in chosen.
            # If we remove p, the pairs involving p are gone.
            # So we need to check if there is a pair (a,b) in chosen\{p} that blocks p.
            # Let's calculate this specifically for p.
            
            p_blocked_by_rest = False
            for a in chosen:
                if a == p:
                    continue
                # Check if there is b in chosen \ {p} such that b in iso_third[a][p]
                # i.e., (a, p, b) is isosceles.
                # Note: iso_third[a][p] contains b if (a,p,b) is isosceles.
                # We need b in chosen and b != p and b != a.
                # Since p is not in chosen\{p}, we just check intersection of (chosen - {p}) and iso_third[a][p].
                if (chosen - {p}) & iso_third[a][p]:
                    p_blocked_by_rest = True
                    break
            
            if not p_blocked_by_rest:
                continue # p cannot be removed, it is not blocked by the rest
            
            # Now check if removing p unblocks any other point q not in chosen
            # We need to simulate the removal of p from block counts.
            # For each a in chosen \ {p}:
            #   For each q in iso_third[p][a]:
            #     block_count[q] -= 1
            #   (Also for b in iso_third[a][p]? No, iso_third is symmetric in the sense that
            #    if (p,a,q) is isosceles, then (a,p,q) is isosceles.
            #    So iso_third[p][a] == iso_third[a][p].)
            
            # We need to check if any q not in chosen (and q != p) has block_count[q] become 0.
            # And also, does p itself have block_count > 0? (We already checked p is blocked by rest).
            
            # Let's store old block counts for affected q's to rollback if needed.
            affected = {}
            valid = True
            
            for a in chosen:
                if a == p:
                    continue
                for q in iso_third[p][a]:
                    if q in affected:
                        affected[q] -= 1
                    else:
                        affected[q] = block_count[q] - 1
                        if q not in chosen and q != p:
                            if affected[q] <= 0:
                                valid = False
                                break
                if not valid:
                    break
            
            if valid:
                # Also need to check if p itself is still "blocked" in the sense of the global count?
                # No, we just need the set to be maximal.
                # The points in the new set are chosen \ {p}.
                # The points outside are (all \ chosen) U {p}.
                # We checked that all q in (all \ chosen) are still blocked (affected[q] > 0).
                # We checked that p is blocked by chosen \ {p}.
                # So it is a valid maximal set.
                
                # Commit the removal
                new_chosen = chosen - {p}
                
                # Update block_count globally
                for q, new_count in affected.items():
                    block_count[q] = new_count
                
                return new_chosen, block_count
            
        return None

    # Main loop
    # We run multiple iterations of: Generate -> Prune
    # We keep the best result found so far.
    
    for _ in range(50): # 50 iterations
        if time.time() - start_time > time_limit:
            break
        
        # 1. Generate initial maximal set
        initial_set = generate_initial_maximal()
        
        # 2. Compute initial block counts
        block_count = [0] * num_points
        # For each pair (a, b) in initial_set:
        #   For each q in iso_third[a][b]:
        #     block_count[q] += 1
        
        # Optimization: Only count for q not in initial_set?
        # No, we need block_count for all q to handle removals correctly.
        # But for q in initial_set, block_count is not directly used for maximality check
        # (since they are in the set), but it is used when we remove a point p,
        # to check if p is blocked by the rest?
        # Actually, for p in initial_set, we check p_blocked_by_rest explicitly.
        # For q not in initial_set, we use block_count[q].
        
        # So we only really need block_count for q NOT in initial_set.
        # But when we remove p, p becomes "not in set".
        # So we need block_count[p] to be accurate?
        # In the removal check, we calculate `p_blocked_by_rest` explicitly.
        # And for other q, we use `block_count[q]`.
        # So we can just maintain block_count for all, it's simpler.
        
        for a in initial_set:
            for b in initial_set:
                if a < b: # Avoid double counting pairs?
                    # No, block_count counts the number of pairs.
                    # Pair (a,b) is the same as (b,a).
                    # In my `iso_third` definition, (a,b,q) is isosceles.
                    # If I iterate a < b, I count each pair once.
                    # Is that correct?
                    # "blocked if there EXISTS a pair".
                    # So count > 0 is what matters.
                    # So counting each unordered pair once is sufficient.
                    for q in iso_third[a][b]:
                        block_count[q] += 1
        
        current_set = initial_set
        current_block_count = block_count
        
        # 3. Prune
        improved = True
        while improved:
            improved = False
            result = try_remove(current_set, current_block_count)
            if result:
                current_set, current_block_count = result
                improved = True
        
        # 4. Update best
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = [all_points[i] for i in current_set]
            
    return best_set
# EVOLVE-BLOCK-END
