"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time to save time during construct()
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

# Precompute for each point, the set of other points at the same distance?
# Actually, we need to check isosceles triangles.
# Triangle (a, b, c) is isosceles if d(a,b)==d(a,c) or d(a,b)==d(b,c) or d(a,c)==d(b,c).
# Note: d(a,a)=0. We only care about distinct points.

# Let's precompute for each point i, and each distance d, the set of points j such that dist2[i,j] == d.
# This helps in checking "d(new, a) == d(new, b)" quickly?
# Or better: For a given set S, a point c is "blocked" if there exist a, b in S such that
# (d(c,a) == d(c,b)) or (d(c,a) == d(a,b)) or (d(c,b) == d(a,b)).

# To make the search fast, we will use a backtracking algorithm that tries to build a set of size K.
# Heuristic: When choosing the next point to add to S, we prefer points that are "compatible" 
# (don't form isosceles with existing S) and that "block" many other points.

def _is_isosceles_with_existing(new_idx, S_indices):
    """
    Check if adding new_idx to S_indices creates any isosceles triangle.
    S_indices is a set or list of indices.
    """
    if len(S_indices) < 2:
        return False
    
    # Check 1: d(new, a) == d(new, b) for any a, b in S
    # Check 2: d(new, a) == d(a, b) for any a, b in S
    # Check 3: d(new, b) == d(a, b) for any a, b in S (symmetric to 2)
    
    # We can optimize this.
    # Let's just iterate. S is small in the early stages of finding a minimal set.
    S_list = list(S_indices)
    n_s = len(S_list)
    
    # Distances from new to all in S
    d_new_to_S = [DIST2[new_idx, s] for s in S_list]
    
    # Check 1: Duplicate distances from new to S
    # If any two are equal, isosceles.
    # Since S is small, sorting or using a set is fine.
    if len(set(d_new_to_S)) < n_s:
        return True
        
    # Check 2 & 3: d(new, a) == d(a, b)
    for i in range(n_s):
        a = S_list[i]
        d_na = d_new_to_S[i]
        for j in range(n_s):
            if i == j:
                continue
            b = S_list[j]
            d_ab = DIST2[a, b]
            if d_na == d_ab:
                return True
            # d(new, b) == d(a, b) is covered when we swap roles in the loop?
            # No, the loop covers pairs (a,b). 
            # Condition: d(new, a) == d(a, b) OR d(new, b) == d(a, b).
            # The loop iterates i, j. 
            # If d(new, a) == d(a, b), we catch it.
            # If d(new, b) == d(a, b), we catch it when i=b, j=a? 
            # Wait, d_ab is symmetric. d(a,b) = d(b,a).
            # In the iteration (i=a, j=b), we check d(new, a) == d(a, b).
            # In the iteration (i=b, j=a), we check d(new, b) == d(b, a).
            # So yes, both cases are covered.
            
    return False

def _count_blocked_points(S_indices):
    """
    Count how many points outside S are blocked by S.
    A point c is blocked if it forms an isosceles triangle with some pair in S.
    """
    S_set = set(S_indices)
    blocked_count = 0
    # We can iterate through all points not in S
    # To speed this up, we can precompute "blocking power" but it depends on S.
    # For small S, this is fast enough.
    for c in range(V):
        if c in S_set:
            continue
        # Check if c is blocked by S
        # c is blocked if exists a, b in S such that (a,b,c) is isosceles.
        # We can reuse the logic from _is_isosceles_with_existing but adapted.
        # Actually, _is_isosceles_with_existing(c, S_indices) returns True if c can NOT be added.
        # Which means c IS blocked.
        if _is_isosceles_with_existing(c, S_indices):
            blocked_count += 1
            
    return blocked_count

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # 10s buffer for safety
    
    # We want to find a minimal maximal set.
    # Let's try to find a set of size K for K = 1, 2, 3...
    # K=1: A single point blocks no one (needs 2 points for a triangle). So not maximal unless V=1.
    # K=2: Two points a, b. A point c is blocked if d(c,a)==d(c,b) or d(c,a)==d(a,b) or d(c,b)==d(a,b).
    #      This is a specific geometric condition.
    # We will use a backtracking search to find a valid set of size K that is maximal.
    
    # To guide the search, we'll use a heuristic:
    # 1. Start with an empty set.
    # 2. At each step, choose the next point from the valid candidates.
    # 3. Prioritize points that maximize the number of blocked points (greedy heuristic for maximality).
    # 4. If we reach size K and the set is maximal, we are done.
    # 5. If we reach size K and the set is NOT maximal, backtrack.
    # 6. If we get stuck (no valid points to add before reaching K), backtrack.
    
    # Optimization: Precompute valid pairs?
    # For K up to ~10, backtracking with good pruning should be fast.
    
    # Let's define a recursive backtracking function.
    # State: current_set (list), blocked_mask (bitmask of points blocked by current_set)
    # Using a bitmask for blocked points will make the "maximal" check very fast (just check if mask covers all non-S points).
    
    # Precompute for each point i, the bitmask of points j such that (i, j) is a "valid pair" (no isosceles with 0 points? No, just distinct).
    # Actually, "valid" depends on the whole set.
    
    # Let's use a bitmask for the set S and a bitmask for Blocked.
    # V=256, so we need 256-bit integers. Python handles big ints natively.
    
    # Precompute for each i, the set of j such that i and j are distinct.
    # And for each pair (i, j), which points c are blocked by (i, j)?
    # c is blocked by (i, j) if (i, j, c) is isosceles.
    # Let's precompute `blocked_by_pair[i][j]` as a bitmask of points c such that (i, j, c) is isosceles.
    # Note: c can be i or j? No, c must be distinct from i and j for a triangle.
    # But if c=i, d(i,i)=0, d(i,j)=d_ij, d(j,i)=d_ij. 0 != d_ij. So not isosceles.
    # So we only care about c != i and c != j.
    
    blocked_by_pair = np.zeros((V, V), dtype=object) # Object to hold big ints
    
    # This precomputation is O(V^3) which is 256^3 ~ 16 million operations.
    # Each operation involves distance checks. Might be slow in Python.
    # Let's optimize.
    # For each i, j:
    #   d_ij = DIST2[i, j]
    #   For each c:
    #      if c == i or c == j: continue
    #      d_ic = DIST2[i, c]
    #      d_jc = DIST2[j, c]
    #      if d_ic == d_jc or d_ic == d_ij or d_jc == d_ij:
    #          blocked_by_pair[i, j] |= (1 << c)
    
    # 16M iterations in Python is too slow (seconds to minutes).
    # We need a faster way or a different strategy.
    
    # Alternative: Don't precompute all pairs. Compute on the fly during backtracking.
    # The backtracking tree will be small if the heuristic is good.
    
    # Let's try a simpler heuristic search:
    # 1. Generate a random valid set of size K (using the greedy add method from before).
    # 2. Check if it's maximal.
    # 3. If not, try to "fix" it by swapping points or adding/removing?
    #    But we can't add (fixed size K). We can swap.
    #    If we swap a point in S with a point not in S, we might break validity or maximality.
    
    # Let's stick to Backtracking but with a very strong pruning:
    # We want to find S of size K.
    # We'll maintain:
    #   S: list of indices
    #   Valid_Candidates: set of points that can be added to S without creating isosceles.
    #   Blocked_Mask: bitmask of points blocked by current S.
    
    # Pruning:
    # 1. If len(S) == K:
    #    Check if Blocked_Mask covers all points not in S.
    #    If yes, return S.
    #    If no, backtrack.
    # 2. If len(S) + len(Valid_Candidates) < K:
    #    Backtrack (not enough candidates to reach size K).
    # 3. If (Blocked_Mask | (1 << all points in S)) does not cover all points, 
    #    and even if we add all remaining Valid_Candidates, they can't block enough?
    #    This is hard to estimate.
    
    # Let's implement the backtracking with a greedy choice order.
    # We'll sort Valid_Candidates by "blocking power" (how many new points they block) in descending order.
    
    def _get_blocking_power(candidate, current_S, current_blocked_mask):
        """
        Estimate how many new points 'candidate' would block if added to current_S.
        """
        # Points blocked by (candidate, s) for s in current_S
        # Plus points blocked by (candidate, s1, s2) for s1, s2 in current_S?
        # No, adding candidate creates triangles (candidate, s, c) for s in S.
        # It also creates triangles (s1, s2, candidate) which are already checked for validity.
        # The NEW blocked points are those c such that (candidate, s, c) is isosceles for some s in S.
        # We can compute this on the fly.
        
        new_blocked = 0
        # We need to know which points are NOT already blocked.
        # It's faster to just compute the bitmask of points blocked by candidate with current_S 
        # and count bits not in current_blocked_mask.
        
        # For each s in current_S:
        #   Find all c such that (candidate, s, c) is isosceles.
        #   This is O(V) per s. Total O(|S|*V).
        
        # Let's just do a rough estimate: 
        # The number of points at distance d(candidate, s) from candidate is small.
        # Let's just return a random value or 0 for now to keep it simple, 
        # or just use the count of valid candidates?
        
        # Actually, a better heuristic: Prioritize points that are "central" or have many neighbors?
        # Let's just try random order first, with backtracking.
        return 0

    # Given the complexity, let's try a different structural approach:
    # **Simulated Annealing on the Set Representation**
    # Start with a valid set (e.g., from greedy).
    # Move: Remove a point, add a different point.
    # Cost: 0 if valid and maximal, else large penalty.
    # But we want MINIMAL size.
    # So we should fix the size K and search for a valid maximal set of that size.
    
    # Let's try K from 1 to 15.
    # For each K, run a local search (Simulated Annealing) to find a valid maximal set of size K.
    
    def _is_valid_maximal(S):
        S_set = set(S)
        # Check validity
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                # Check if S[i], S[j] form isosceles with any other in S?
                # No, we check all triples.
                pass # This is O(K^3). K is small.
        
        # Better: Use the _is_isosceles_with_existing logic incrementally?
        # Just check all triples in S.
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    d_ab = DIST2[a, b]
                    d_ac = DIST2[a, c]
                    d_bc = DIST2[b, c]
                    if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                        return False, False # Invalid
        
        # Check maximality
        # Every point not in S must be blocked.
        for c in range(V):
            if c in S_set:
                continue
            # Check if c is blocked by S
            # c is blocked if exists a, b in S such that (a, b, c) is isosceles.
            blocked = False
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    a, b = S[i], S[j]
                    d_ab = DIST2[a, b]
                    d_ac = DIST2[a, c]
                    d_bc = DIST2[b, c]
                    if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return True, False # Valid but not maximal
        
        return True, True # Valid and maximal

    best_solution = None
    best_size = V
    
    # We will try sizes from 1 upwards.
    # For each size, we run a few random restarts of a "Grow and Swap" heuristic.
    
    for K in range(1, 15):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random configurations of size K
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Start with a random valid set of size K?
            # Hard to generate directly.
            # Instead, start with a valid set of size K from a greedy build (might fail if greedy gets stuck early).
            # Let's build a valid set of size K using greedy.
            # If greedy fails to reach K, restart.
            
            # Greedy build of size K
            current_S = []
            order = list(range(V))
            random.shuffle(order)
            
            for idx in order:
                if len(current_S) == K:
                    break
                if _is_isosceles_with_existing(idx, current_S):
                    continue
                current_S.append(idx)
                
            if len(current_S) < K:
                continue
                
            # Now we have a valid set of size K.
            # Check if it's maximal.
            is_valid, is_maximal = _is_valid_maximal(current_S)
            
            if is_maximal:
                found = True
                best_size = K
                best_solution = current_S
                break # Found a maximal set of size K, which is minimal for this K.
                # Since we iterate K upwards, the first found is the global minimum.
            
            # If not maximal, we could try to swap points to make it maximal.
            # But that's complex. Let's just rely on more restarts.
            # Or, we can try to "fix" it by swapping a point that doesn't contribute to blocking 
            # with a point that does.
            
            # Simple Swap Heuristic:
            # 1. Identify points in S that are "redundant" for blocking? Hard.
            # 2. Just try random swaps.
            for swap_iter in range(50):
                if time.time() > deadline:
                    break
                    
                # Pick a random point in S to remove
                i = random.randint(0, K-1)
                old_point = current_S[i]
                
                # Pick a random point not in S to add
                candidates = [p for p in range(V) if p not in current_S]
                if not candidates:
                    break
                new_point = random.choice(candidates)
                
                # Try the swap
                new_S = current_S[:]
                new_S[i] = new_point
                
                # Check validity
                valid, maximal = _is_valid_maximal(new_S)
                if valid and maximal:
                    found = True
                    best_size = K
                    best_solution = new_S
                    break
                elif valid:
                    # Keep the valid swap even if not maximal? 
                    # Yes, it might be a stepping stone.
                    current_S = new_S
            
            if found:
                break
                
        if found:
            break # Found minimal size, stop searching larger K
            
    if best_solution is None:
        # Fallback: return a valid maximal set from a standard greedy run
        current_S = []
        for idx in range(V):
            if _is_isosceles_with_existing(idx, current_S):
                continue
            current_S.append(idx)
        best_solution = current_S
        
    return [POINTS[i] for i in best_solution]

# EVOLVE-BLOCK-END
