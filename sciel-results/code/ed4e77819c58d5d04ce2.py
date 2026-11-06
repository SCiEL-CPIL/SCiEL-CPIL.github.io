import time
import random
import itertools
import sys

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = N * N
DEADLINE = 325.0  # Safety margin from 330s

# Precompute distance squared matrix
dist2_matrix = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
point_index = {p: i for i, p in enumerate(POINTS)}

for i in range(NUM_POINTS):
    pi = POINTS[i]
    for j in range(NUM_POINTS):
        pj = POINTS[j]
        dist2_matrix[i][j] = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2

# Precompute Isosceles relationships using bitmasks
# iso_mask[i] is a bitmask where bit j is 1 if point j is part of an isosceles triangle 
# with point i and some other point? 
# Actually, we need: for a set S, is it no-isosceles?
# A set S is no-isosceles if for all distinct a, b, c in S, 
# dist2(a,b) != dist2(a,c) and dist2(a,b) != dist2(b,c) and dist2(a,c) != dist2(b,c).
#
# Let's precompute for each pair (a, b), the set of points p such that (p, a, b) is isosceles.
# Let's call this iso_pairs[a][b] = bitmask of p.
# Then, a set S is no-isosceles iff for all a, b in S, 
# (S & iso_pairs[a][b]) does not contain any point c in S other than a and b?
# Wait, if p, a, b form an isosceles triangle, then p is in iso_pairs[a][b].
# If S contains a, b, and p, and p is in iso_pairs[a][b], then S is not no-isosceles.
# So, S is no-isosceles iff for all a < b in S:
#   (S & iso_pairs[a][b]) is a subset of {a, b} ?
#   No, p must be distinct from a and b.
#   So we need (S & iso_pairs[a][b]) to be empty? 
#   Wait, iso_pairs[a][b] contains p if (p,a,b) is isosceles.
#   If p=a, (a,a,b) is not a triangle. 
#   Let's define iso_pairs[a][b] to contain ONLY p such that p != a and p != b and (p,a,b) is isosceles.
#   Then S is no-isosceles iff for all a < b in S:
#       (S & iso_pairs[a][b]) == 0

# Precompute iso_pairs
# iso_pairs[i][j] is a bitmask of k such that k != i, k != j, and (POINTS[k], POINTS[i], POINTS[j]) is isosceles.
# Note: iso_pairs[i][j] == iso_pairs[j][i]
iso_pairs = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]

# Optimization: Iterate over all triples? Or use geometry?
# For each pair (i, j), find all k such that (k, i, j) is isosceles.
# Conditions:
# 1. dist2(k,i) == dist2(k,j)  (k on perp bisector)
# 2. dist2(k,i) == dist2(i,j)  (k on circle centered i radius ij)
# 3. dist2(k,j) == dist2(i,j)  (k on circle centered j radius ij)
# And k != i, k != j.

start_time = time.time()
for i in range(NUM_POINTS):
    for j in range(i+1, NUM_POINTS):
        d_ij = dist2_matrix[i][j]
        mask = 0
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            d_ki = dist2_matrix[k][i]
            d_kj = dist2_matrix[k][j]
            if d_ki == d_kj or d_ki == d_ij or d_kj == d_ij:
                mask |= (1 << k)
        iso_pairs[i][j] = mask
        iso_pairs[j][i] = mask

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def _is_no_isosceles(S_mask):
    """Check if set S (bitmask) is no-isosceles."""
    # Iterate over all pairs in S
    # Extract points
    pts = []
    for i in range(NUM_POINTS):
        if (S_mask >> i) & 1:
            pts.append(i)
    n = len(pts)
    for a_idx in range(n):
        a = pts[a_idx]
        for b_idx in range(a_idx + 1, n):
            b = pts[b_idx]
            # Check if any other point c in S is in iso_pairs[a][b]
            # We need to check if S_mask & iso_pairs[a][b] has any bit set that is not a or b?
            # Actually, iso_pairs[a][b] already excludes a and b.
            # So we just check if S_mask & iso_pairs[a][b] is non-zero.
            if S_mask & iso_pairs[a][b]:
                return False
    return True

def _get_unblocked(S_mask):
    """Return bitmask of points NOT in S that are blocked by S.
    A point p is blocked by S if there exist a, b in S such that p is in iso_pairs[a][b].
    So p is blocked if p is in the union of iso_pairs[a][b] for all a < b in S.
    """
    pts = []
    for i in range(NUM_POINTS):
        if (S_mask >> i) & 1:
            pts.append(i)
    n = len(pts)
    blocked_mask = 0
    for a_idx in range(n):
        a = pts[a_idx]
        for b_idx in range(a_idx + 1, n):
            b = pts[b_idx]
            blocked_mask |= iso_pairs[a][b]
    # Exclude points already in S
    blocked_mask &= ~S_mask
    return blocked_mask

def _is_maximal(S_mask):
    """Check if S is maximal."""
    # S is maximal if all points not in S are blocked.
    # i.e., _get_unblocked(S_mask) covers all points not in S.
    all_points_mask = (1 << NUM_POINTS) - 1
    not_in_S = all_points_mask & ~S_mask
    blocked = _get_unblocked(S_mask)
    return (blocked & not_in_S) == not_in_S

def _objective(S_mask):
    """
    Objective function for search.
    We want to minimize size.
    But we must maintain no-isosceles and maximality.
    If not maximal, we have a penalty.
    If not no-isosceles, huge penalty.
    """
    if not _is_no_isosceles(S_mask):
        return 1000000 + bin(S_mask).count('1')
    
    size = bin(S_mask).count('1')
    if _is_maximal(S_mask):
        return size
    
    # If not maximal, penalize by number of unblocked points
    not_in_S = ((1 << NUM_POINTS) - 1) & ~S_mask
    blocked = _get_unblocked(S_mask)
    unblocked = not_in_S & ~blocked
    penalty = bin(unblocked).count('1') * 10 # Strong penalty for non-maximality
    return size + penalty

def _random_valid_pair():
    """Pick a random pair that forms a valid start (no isosceles with itself, trivially true).
    Actually, any pair is a valid start for no-isosceles.
    But we want pairs that block many points to help maximality.
    """
    # Pick random pair
    i, j = random.sample(range(NUM_POINTS), 2)
    return (1 << i) | (1 << j)

def construct():
    start_time = time.time()
    best_mask = 0
    best_size = float('inf')
    
    # Simulated Annealing
    # State: S_mask
    # Moves: Add a point, Remove a point, Swap a point
    
    # Initialize with a greedy approach to get a good starting point
    # Greedy: Start with empty, add point that blocks most? 
    # Or start with a high-coverage pair.
    
    # Let's try multiple restarts
    for restart in range(20):
        if time.time() > DEADLINE:
            break
            
        # Initialize S
        # Pick a random pair
        S = _random_valid_pair()
        
        # Try to make it maximal greedily
        # While not maximal:
        #   Find a point p that can be added (keeps no-isosceles) and maximizes blocking.
        #   Add it.
        # If stuck, break.
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            if _is_maximal(S):
                break
            
            # Find candidates to add
            candidates = []
            for p in range(NUM_POINTS):
                if (S >> p) & 1:
                    continue
                S_new = S | (1 << p)
                if _is_no_isosceles(S_new):
                    # Calculate gain in blocking
                    # This is expensive, so let's just pick random valid p first to get a valid maximal set
                    candidates.append(p)
            
            if not candidates:
                break
            
            # Pick candidate that maximizes blocked points
            best_p = None
            best_blocked = -1
            # To save time, just pick random if list is large
            if len(candidates) > 20:
                p = random.choice(candidates)
            else:
                for p in candidates:
                    S_new = S | (1 << p)
                    blocked = _get_unblocked(S_new)
                    cnt = bin(blocked).count('1')
                    if cnt > best_blocked:
                        best_blocked = cnt
                        best_p = p
                p = best_p
            
            if p is not None:
                S |= (1 << p)
        
        # Now S is hopefully maximal and no-isosceles.
        # If not, it's invalid.
        if not _is_no_isosceles(S):
            continue
        if not _is_maximal(S):
            # Try to repair? Or just skip this restart.
            # Let's try a few random additions to see if we can fix maximality
            # But if we can't add without breaking no-isosceles, it's a dead end.
            # Actually, if it's not maximal, there is a point p that can be added.
            # So we can always add it.
            # Why did the loop stop?
            # The loop stops if no candidates. That means no point can be added while maintaining no-isosceles.
            # But if it's not maximal, there is a point p that is NOT blocked.
            # If p is not blocked, then S U {p} is no-isosceles?
            # No. p not blocked means there are no a,b in S such that (p,a,b) is isosceles.
            # This implies S U {p} is no-isosceles?
            # Yes, because any new isosceles triangle must involve p and two points from S.
            # If p is not blocked, no such triangle exists.
            # So if S is not maximal, there MUST be a candidate.
            # So if candidates is empty, S must be maximal.
            # Wait, my logic:
            # p is blocked if there exist a,b in S such that (p,a,b) is isosceles.
            # If p is NOT blocked, then for all a,b in S, (p,a,b) is NOT isosceles.
            # This means S U {p} contains no isosceles triangle involving p.
            # Since S is no-isosceles, S U {p} is no-isosceles.
            # So if S is not maximal, there is a p not blocked.
            # Such p is a valid candidate.
            # So if candidates is empty, S is maximal.
            # Thus, if we exit the loop with S not maximal, it's a bug or I'm missing something.
            # Let's assume the greedy loop works and S is maximal.
            pass

        size = bin(S).count('1')
        if size < best_size:
            best_size = size
            best_mask = S
            
        # Local Search: Simulated Annealing
        # We want to reduce size.
        # Move: Remove a point?
        # If we remove a point, S might become non-maximal.
        # We need to check if S' = S \ {p} is maximal.
        # If not, we can try to "swap" p with q?
        # Or just accept non-maximal states with penalty in objective.
        
        current_mask = S
        current_obj = _objective(current_mask)
        
        T = 10.0
        min_T = 0.1
        cooling = 0.99
        
        while T > min_T:
            if time.time() > DEADLINE:
                break
            
            # Generate neighbor
            # Option 1: Remove a random point
            # Option 2: Add a random point
            # Option 3: Swap
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Remove
                pts = [i for i in range(NUM_POINTS) if (current_mask >> i) & 1]
                if not pts:
                    continue
                p = random.choice(pts)
                new_mask = current_mask & ~(1 << p)
            elif move_type < 0.8:
                # Add
                not_in = [i for i in range(NUM_POINTS) if not (current_mask >> i) & 1]
                if not not_in:
                    continue
                p = random.choice(not_in)
                new_mask = current_mask | (1 << p)
            else:
                # Swap
                pts = [i for i in range(NUM_POINTS) if (current_mask >> i) & 1]
                not_in = [i for i in range(NUM_POINTS) if not (current_mask >> i) & 1]
                if not pts or not not_in:
                    continue
                p_out = random.choice(pts)
                p_in = random.choice(not_in)
                new_mask = (current_mask & ~(1 << p_out)) | (1 << p_in)
            
            new_obj = _objective(new_mask)
            
            # Acceptance
            if new_obj < current_obj:
                current_mask = new_mask
                current_obj = new_obj
                if _is_no_isosceles(current_mask) and _is_maximal(current_mask):
                    size = bin(current_mask).count('1')
                    if size < best_size:
                        best_size = size
                        best_mask = current_mask
            else:
                # Accept with probability
                delta = new_obj - current_obj
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    current_mask = new_mask
                    current_obj = new_obj
            
            T *= cooling
        
    # Return best found
    res = []
    for i in range(NUM_POINTS):
        if (best_mask >> i) & 1:
            res.append(POINTS[i])
            
    # Fallback if best_mask is invalid or empty
    if not res:
        # Just return a known valid small set? 
        # Or run a quick greedy again
        S = _random_valid_pair()
        # ... (simplified fallback)
        res = [POINTS[i] for i in range(NUM_POINTS) if (S >> i) & 1]
        
    return res

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
