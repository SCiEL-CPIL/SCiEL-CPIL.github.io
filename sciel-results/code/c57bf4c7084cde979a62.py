# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 620.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES = p*p + p*p*p  # 49 + 343 = 392? No.
# Directions: v=(0,1) -> 49 lines. v=(1,m) for 49 m values -> 49*49 = 2401 lines.
# Total lines = 49 + 2401 = 2450.
N_LINES_TOTAL = 2450
N_PER_LINE = p*p  # 49

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Unital condition: 2*xa + ya^2 - yb^2*K = 0 mod 7
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        # y is special if ya is NOT a square mod 7
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _precompute_lines():
    """
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            for k in range(N_PER_LINE):
                ya = k // p
                yb = k % p
                lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
            line_idx += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (ca, cb) - the y-intercept essentially
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        # y = c + x*m
                        # ya = ca + ma*xa + mb*xb*alpha^2 = ca + ma*xa + 3*mb*xb
                        # yb = cb + ma*xb + mb*xa
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    
    return lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Generate initial S from seed
    in_S = _generate_seed_excluded().astype(np.int8)
    
    # Precompute lines
    lines = _precompute_lines()
    
    # Compute initial bad counts
    # bad[x] = count of lines L containing x such that |S ∩ (L \ {x})| > 0
    # This is equivalent to: for each line L, if |S ∩ L| > (1 if x in S else 0), then bad[x] += 1
    
    # Vectorized approach to compute initial bad counts:
    # For each line, count S points.
    # line_s_counts[i] = sum(in_S[lines[i, :]])
    line_s_counts = np.sum(in_S[lines], axis=1)  # shape (N_LINES_TOTAL,)
    
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # For each line i, for each point x in lines[i]:
    #   if line_s_counts[i] > (1 if in_S[x] else 0): bad[x] += 1
    # We can do this in a loop over lines, but vectorize over points within a line.
    # Or use a scatter-add approach.
    
    # Let's try a loop over lines, which is 2450 iterations. Inside, vectorize over 49 points.
    for i in range(N_LINES_TOTAL):
        cnt = line_s_counts[i]
        if cnt > 0:
            line_pts = lines[i, :]
            # Condition: cnt > in_S[line_pts]
            # Note: in_S[line_pts] is 1 if point is in S, 0 otherwise.
            # We want to add 1 to bad[x] for x where cnt > in_S[x].
            mask = cnt > in_S[line_pts]
            bad[line_pts[mask]] += 1
            
    # Greedy removal: try to remove points from S.
    # A point c can be removed if for all lines L containing c, the new condition
    # (cnt_L - 1) > (1 if x in S_new else 0) does not cause bad[x] to drop below 0?
    # No, the condition for validity is that NO point x has bad[x] > 0?
    # Wait, the Nikodym condition is: for every point x, there exists a line L_x through x
    # such that L_x \ {x} is entirely in N.
    # This means for every x, there is at least one direction v such that S ∩ (L_x,v \ {x}) = empty.
    # Equivalently, for every x, there is at least one direction v such that |S ∩ L_x,v| <= (1 if x in S else 0).
    # So bad[x] (count of directions where S ∩ (L \ {x}) is non-empty) must be < 50 (total directions).
    # Actually, bad[x] must be <= 49. If bad[x] == 50, then for ALL directions, the punctured line hits S.
    # So x is NOT covered.
    # Thus, the validity condition is: max(bad) <= 49.
    
    # When we remove c from S:
    # For each line L containing c:
    #   OLD cnt = line_s_counts[L]. NEW cnt = OLD cnt - 1.
    #   For each x in L:
    #     OLD bad_contrib = 1 if OLD cnt > (1 if x in S else 0) else 0
    #     NEW bad_contrib = 1 if NEW cnt > (1 if x in S_new else 0) else 0
    #     Note: x in S_new is same as x in S, except x=c where it becomes 0.
    
    # Let's analyze the change in bad[x] for each x in L when removing c.
    # Let s_x = 1 if x in S (OLD), else 0.
    # Let s_c = 1 (c is in S).
    # OLD cnt = C. NEW cnt = C - 1.
    
    # Case x = c:
    #   OLD: s_c = 1. Condition: C > 1.
    #   NEW: s_c_new = 0. Condition: C - 1 > 0.
    #   Change:
    #     If C > 1: OLD=1, NEW=1 (if C-1>0 i.e. C>1). No change.
    #     If C == 1: OLD=0 (1>1 false), NEW=0 (0>0 false). No change.
    #     If C == 0: Impossible since c in S.
    #   So bad[c] does not change?
    #   Let's check C=2. OLD: 2>1 (True). NEW: 1>0 (True). No change.
    #   Let's check C=1. OLD: 1>1 (False). NEW: 0>0 (False). No change.
    #   So bad[c] is unchanged.
    
    # Case x != c:
    #   s_x is unchanged.
    #   OLD: C > s_x.
    #   NEW: C - 1 > s_x.
    #   Change:
    #     If C > s_x and C - 1 <= s_x: bad[x] decreases by 1.
    #     This happens if s_x = C - 1 and C > s_x (which is true since C > C-1).
    #     So if s_x = C - 1, bad[x] decreases by 1.
    #     Note: s_x is 0 or 1.
    #     If s_x = 0: C - 1 = 0 => C = 1.
    #     If s_x = 1: C - 1 = 1 => C = 2.
    
    # So, removing c from S:
    # For each line L containing c with count C:
    #   If C == 1:
    #     For all x in L with x != c and x not in S (s_x=0), bad[x] decreases by 1.
    #   If C == 2:
    #     For all x in L with x != c and x in S (s_x=1), bad[x] decreases by 1.
    #   If C > 2:
    #     No changes.
    #   If C < 1: Impossible.
    
    # We want to remove c if it leads to a valid state (max bad <= 49).
    # Since removing c only DECREASES bad counts (or keeps them same), it can never make the set INVALID.
    # It can only make it MORE valid (or keep it valid).
    # Wait, is that true?
    # bad[x] = count of "bad" directions.
    # If we remove a point from S, we are making S smaller.
    # Smaller S means fewer intersections with S.
    # So bad[x] can only decrease or stay the same.
    # Therefore, if the initial set is valid (max bad <= 49), any subset is also valid.
    # The seed set is likely valid.
    # So we can remove ANY point from S?
    # If so, we can remove ALL points from S, making S empty and N = F_49^2.
    # Is S empty a valid Nikodym set?
    # If S is empty, then for any x, and any line L through x, L \ {x} is entirely in N (since N is everything).
    # So yes, S empty is valid.
    # Why does the seed program not just return the full set?
    # "Metric (higher is better): the complement size 2401 - |N|, the number of distinct points not in N."
    # Wait. The metric is the size of the COMPLEMENT of N, which is |S|.
    # "Score is proportional to it."
    # So we want to MAXIMIZE |S|?
    # "The complement size 2401 - |N| ... score is proportional to it."
    # If N is the Nikodym set, S is the complement.
    # We want to maximize the size of S?
    # Let's re-read carefully.
    # "Metric (higher is better): the complement size 2401 - |N|, the number of distinct points not in N."
    # Yes, we want to MAXIMIZE the number of points NOT in N.
    # So we want S to be as LARGE as possible.
    # The current program tries to ADD points to S.
    # My previous analysis of "removing" was backwards.
    # We want to add points to S such that the Nikodym condition remains satisfied.
    # The Nikodym condition is: for all x, there exists a line L_x such that L_x \ {x} subset N.
    # Equivalently: for all x, there exists a direction v such that S ∩ (L_x,v \ {x}) = empty.
    # Equivalently: for all x, bad[x] < 50.
    
    # So the constraint is: max(bad) <= 49.
    # We start with a seed S.
    # We want to add points to S.
    # Adding a point c to S increases bad[x] for some x.
    # We must ensure that after adding c, max(bad) <= 49.
    
    # This is exactly what the current program does!
    # It greedily adds points to S if they don't cause any bad[x] to reach 50.
    
    # So the strategy is: Greedy addition of points to S.
    # To improve, we need a better greedy heuristic or a different search.
    # The current program iterates through candidates in a fixed order.
    # A better strategy:
    # 1. Compute the "safety margin" for each candidate: min(49 - bad[x]) over all x that would be affected.
    #    Actually, adding c affects bad[x] for x on lines through c.
    #    The "risk" of adding c is the maximum increase in bad[x] for any x.
    #    If we add c, bad[x] increases by 1 for specific x.
    #    We need to ensure that for all affected x, bad[x] + 1 <= 49, i.e., bad[x] <= 48.
    #    So we can add c if and only if for all x that would have bad[x] incremented, bad[x] <= 48.
    #    Equivalently, min(49 - bad[x]) >= 1 for all affected x.
    #    Or simply: max(bad[x] for affected x) <= 48.
    
    # To prioritize which point to add first:
    # We want to add points that are "safest".
    # A point c is safer if the points x it affects have lower bad[x] values.
    # Specifically, we want to avoid affecting points with high bad[x].
    # So, for each candidate c, compute:
    #   risk(c) = max( bad[x] for x such that bad[x] would be incremented if c is added )
    #   If risk(c) <= 48, c is addable.
    #   Among addable points, pick the one with the LOWEST risk(c).
    #   Or maybe pick the one with the lowest MAX bad[x] among affected points.
    
    # This requires computing the affected points for each candidate.
    # Affected points for c:
    #   For each line L through c:
    #     If cnt_L == 0: affected are x in L \ {c} with x not in S.
    #     If cnt_L == 1: affected are x in L \ {c} with x in S.
    #     If cnt_L > 1: no affected points.
    
    # This is expensive to compute for all 2401 candidates.
    # But we can do it iteratively.
    
    # Alternative:
    # Just try to add points in order of increasing "current bad count of the point itself"?
    # Or random order?
    # The current program uses random order.
    
    # Let's try a priority queue approach.
    # For each candidate c, estimate the risk.
    # Risk is roughly the maximum bad[x] among affected x.
    # If we approximate affected x as just c itself? No, c's bad count doesn't change.
    # The affected x are neighbors of c in the "bad count" graph.
    
    # Let's try a simpler heuristic:
    # Sort candidates by their current bad[x] value (ascending).
    # Points with low bad[x] are "safer" to be near? No, we are adding c, not x.
    # Adding c increases bad[x] for neighbors x.
    # We want to add c if its neighbors have low bad[x].
    # So we want c to have neighbors with low bad[x].
    # This suggests looking at the neighborhood of c.
    
    # Let's implement the "lowest risk first" greedy.
    # 1. Initialize a priority queue with all candidates not in S.
    #    Key: estimated risk.
    #    To estimate risk, we need to know which x will be affected.
    #    This depends on the current state of S (line counts).
    #    So the risk changes as we add points.
    #    We need to update the risks of other candidates when we add a point.
    #    This is complex.
    
    # Simpler:
    # Just do multiple passes.
    # In each pass, iterate through all candidates.
    # If a candidate can be added, add it.
    # Repeat until no more can be added.
    # The order in the pass can be randomized or sorted.
    
    # The current program does ONE pass.
    # Let's do MULTIPLE passes until no progress is made.
    # This is a standard greedy local search improvement.
    
    # Optimization:
    # Instead of checking all candidates from scratch, we can maintain a list of "active" candidates.
    # But simple multiple passes might be enough.
    
    # Let's implement multiple passes.
    
    np.random.seed(42)
    
    progress = True
    while progress:
        progress = False
        # Shuffle candidates to avoid bias
        candidates = [i for i in range(N_TOTAL) if not in_S[i]]
        np.random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 30:
                break
            
            can_add = True
            
            # Check if c_idx can be added
            for l_idx in range(N_LINES_TOTAL):
                # Check if c_idx is on line l_idx
                # This is O(49) per line, 2450 lines -> ~120k ops per candidate.
                # Too slow.
                # We need the list of lines containing c_idx.
                
            # We need point_lines as before.
            # Let's precompute point_lines.
            
            # (Re-adding point_lines computation)
            # point_lines = [[] for _ in range(N_TOTAL)]
            # for i in range(N_LINES_TOTAL):
            #     for idx in lines[i]:
            #         point_lines[idx].append(i)
            
            # This was in the original code. Let's keep it.
            
            # The original code did this.
            # I will restructure to use multiple passes.
            
            # Since I can't easily restructure the whole code in my head right now without risking bugs,
            # let's stick to the original structure but add a loop for multiple passes.
            
                # Actually, the original code is a single pass.
                # Let's wrap the candidate loop in a while loop.
                
                # I will modify the code to do this.
                
                # For now, let's just output the code with multiple passes.
                
                # I'll break out of this inner loop logic and rewrite the function properly below.
                break
        if time.time() > _DEADLINE - 30:
            break

    # This draft is messy. Let's write the clean version.
    
    # Clean Version Plan:
    # 1. Precompute lines and point_lines.
    # 2. Initialize S, line_s_counts, bad.
    # 3. While True:
    #    a. progress = False
    #    b. candidates = [i for i in range(N_TOTAL) if not in_S[i]]
    #    c. shuffle(candidates)
    #    d. for c in candidates:
    #       - if time limit: break
    #       - check if c can be added (using point_lines[c])
    #       - if yes: add c, update line_s_counts, bad, progress=True
    #    e. if not progress: break
    
    # This is O(Passes * Candidates * Lines_per_point * Line_length)
    # Passes might be small (2-5).
    # Candidates ~ 2400.
    # Lines_per_point = 50.
    # Line_length = 49.
    # Ops ~ 5 * 2400 * 50 * 49 ~ 30 million.
    # Python might be slow. 30M simple ops ~ 1-3 seconds.
    # Should be fine within 600s.
    
    # Let's implement this.
    
    # Re-initialize for the clean version
    in_S = _generate_seed_excluded().astype(np.int8)
    lines = _precompute_lines()
    
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    line_s_counts = np.sum(in_S[lines], axis=1)
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    for i in range(N_LINES_TOTAL):
        cnt = line_s_counts[i]
        if cnt > 0:
            line_pts = lines[i, :]
            mask = cnt > in_S[line_pts]
            bad[line_pts[mask]] += 1
            
    np.random.seed(42)
    
    while True:
        if time.time() > _DEADLINE - 30:
            break
            
        progress = False
        candidates = [i for i in range(N_TOTAL) if not in_S[i]]
        np.random.shuffle(candidates)
        
        for c_idx in candidates:
            if time.time() > _DEADLINE - 10:
                break
                
            can_add = True
            
            for l_idx in point_lines[c_idx]:
                cnt = line_s_counts[l_idx]
                
                if cnt == 0:
                    # Affected: x in line, x != c, x not in S
                    # Check if any has bad[x] >= 49
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and not in_S[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                elif cnt == 1:
                    # Affected: x in line, x != c, x in S
                    line_pts = lines[l_idx, :]
                    for x in line_pts:
                        if x != c_idx and in_S[x]:
                            if bad[x] >= 49:
                                can_add = False
                                break
                    if not can_add:
                        break
                # else cnt > 1: no effect
                
            if can_add:
                in_S[c_idx] = 1
                progress = True
                
                for l_idx in point_lines[c_idx]:
                    cnt = line_s_counts[l_idx]
                    line_s_counts[l_idx] += 1
                    line_pts = lines[l_idx, :]
                    
                    if cnt == 0:
                        for x in line_pts:
                            if x != c_idx and not in_S[x]:
                                bad[x] += 1
                    elif cnt == 1:
                        for x in line_pts:
                            if x != c_idx and in_S[x]:
                                bad[x] += 1
                                break # Only one such point
                    
        if not progress:
            break
            
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
