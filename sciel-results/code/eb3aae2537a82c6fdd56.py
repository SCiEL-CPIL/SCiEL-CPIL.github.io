# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401

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
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Unital condition: 2*xa + ya^2 - yb^2*K = 0 mod 7
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        # y is special if ya is a square mod 7
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _precompute_lines():
    """
    Precompute the 2450 lines.
    Returns a list of arrays, each of length 49, containing the indices of points on that line.
    Also returns the 'owner' of each line (the point that defines it uniquely in the context of bad counts? 
    No, bad counts are per-point. We need to know for each point, which lines it is on.
    Actually, for the greedy addition, we need to update bad[x] for all x on the line through c.
    So we need: for each point c, the list of lines that contain c.
    Then for each such line, the list of OTHER points on that line.
    """
    lines = []
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    line_pts.append(_point_to_idx(xa, xb, ya, yb))
            lines.append(np.array(line_pts, dtype=np.int32))
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (c_a, c_b)
            for ca in range(p):
                for cb in range(p):
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            line_pts.append(_point_to_idx(xa, xb, ya, yb))
                    lines.append(np.array(line_pts, dtype=np.int32))
    
    return lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Generate initial S from seed
    in_S = _generate_seed_excluded()
    
    # Precompute lines
    lines = _precompute_lines()
    
    # Compute initial bad counts
    # bad[x] = count of lines L containing x such that (count_S(L) - (1 if x in S else 0)) > 0
    # i.e., count_S(L) > (1 if x in S else 0)
    
    # For each line, count S points.
    # line_s_counts[i] = number of points in lines[i] that are in S.
    line_s_counts = np.array([np.sum(in_S[line]) for line in lines], dtype=np.int32)
    
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # For each line, update bad for its points.
    for i, line in enumerate(lines):
        cnt = line_s_counts[i]
        if cnt > 0:
            for idx in line:
                # x is bad in this direction if there is another S point on the line.
                # This happens if cnt > (1 if in_S[idx] else 0)
                if cnt > (1 if in_S[idx] else 0):
                    bad[idx] += 1
    
    # Greedy: try to add points to S.
    # A point c can be added if for all lines L containing c, the new count_S(L) does not cause
    # any point x on L to have bad[x] == 50.
    # Adding c to S increases line_s_counts[L] by 1 for all L containing c.
    # This increases bad[x] by 1 for all x on L such that the new condition holds.
    # Specifically, if x is on L:
    #   Before: bad_contrib = 1 if cnt > (1 if x in S else 0) else 0
    #   After:  bad_contrib = 1 if cnt+1 > (1 if x in S else 0) else 0
    #   Change: +1 if (cnt+1 > (1 if x in S else 0)) and (cnt <= (1 if x in S else 0))
    #           i.e., if cnt == (1 if x in S else 0)
    
    # So, adding c to S increases bad[x] by 1 for all x on L (L containing c) 
    # such that cnt_L == (1 if x in S else 0).
    
    # We need to ensure that for all such x, bad[x] + 1 < 50, i.e., bad[x] < 49.
    
    # To do this efficiently:
    # For each candidate c:
    #   Find all lines L containing c.
    #   For each such L:
    #      Let cnt = line_s_counts[L].
    #      The points x on L that will have bad[x] incremented are those where:
    #         if x in S: cnt == 1
    #         if x not in S: cnt == 0
    #      We need to check if any such x has bad[x] >= 49.
    
    # Precompute for each point c, the list of lines containing it.
    # And for each line, the list of points.
    
    point_lines = [[] for _ in range(N_TOTAL)]
    for i, line in enumerate(lines):
        for idx in line:
            point_lines[idx].append(i)
    
    # Convert to numpy arrays for speed? Maybe not necessary, lists are fine.
    
    # Candidates: all points not in S.
    candidates = [i for i in range(N_TOTAL) if not in_S[i]]
    
    # Shuffle candidates? Random order might help or hurt. Let's try random.
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    added = 0
    for c_idx in candidates:
        if time.time() > _DEADLINE - 10:
            break
        
        can_add = True
        
        # Get lines containing c_idx
        for l_idx in point_lines[c_idx]:
            line = lines[l_idx]
            cnt = line_s_counts[l_idx]
            
            # Determine which points on this line will have bad incremented.
            # Condition: cnt == (1 if x in S else 0)
            
            # We need to check if any x in line satisfies:
            #   1. (in_S[x] == 1 and cnt == 1) OR (in_S[x] == 0 and cnt == 0)
            #   2. bad[x] >= 49
            
            # If cnt == 0:
            #   We check points x in line where in_S[x] == 0.
            #   If any such x has bad[x] >= 49, we cannot add c.
            # If cnt == 1:
            #   We check points x in line where in_S[x] == 1.
            #   If any such x has bad[x] >= 49, we cannot add c.
            # If cnt > 1:
            #   No points will have bad incremented (since cnt > 1 > 0 and cnt > 1 >= 1).
            #   Wait: if cnt > 1, then for x not in S, cnt > 0 is true, so bad was already incremented.
            #   Adding c makes cnt+1 > 0, still true. No change.
            #   For x in S, cnt > 1 implies cnt > 1, so bad was already incremented.
            #   Adding c makes cnt+1 > 1, still true. No change.
            #   So if cnt > 1, bad counts do NOT change for any x on this line.
            
            if cnt == 0:
                # Check points in line not in S
                for x in line:
                    if not in_S[x]:
                        if bad[x] >= 49:
                            can_add = False
                            break
                if not can_add:
                    break
            elif cnt == 1:
                # Check points in line in S
                for x in line:
                    if in_S[x]:
                        if bad[x] >= 49:
                            can_add = False
                            break
                if not can_add:
                    break
            # else cnt > 1: no change to bad
        
        if can_add:
            # Add c_idx to S
            in_S[c_idx] = 1
            added += 1
            
            # Update line_s_counts
            # And update bad counts for affected points.
            for l_idx in point_lines[c_idx]:
                line = lines[l_idx]
                cnt = line_s_counts[l_idx]
                
                # Increment line_s_counts
                line_s_counts[l_idx] += 1
                
                # Update bad counts
                if cnt == 0:
                    # Points x in line not in S (including c_idx? No, c_idx is now in S)
                    # Wait, the condition was based on OLD state.
                    # Old state: c_idx not in S.
                    # The points that get incremented are those x where:
                    #   OLD: (in_S[x]==1 and cnt==1) or (in_S[x]==0 and cnt==0)
                    #   Here cnt was 0. So we increment x where OLD in_S[x]==0.
                    #   Now c_idx is in S. So we increment all x in line where x != c_idx AND in_S[x]==0 (old).
                    #   Which is all x in line where x != c_idx AND not in_S[x] (current, since only c_idx changed).
                    #   Wait, c_idx was not in S, so it is in the set {x | not in_S[x]}.
                    #   Does c_idx's bad count increase?
                    #   For c_idx: OLD in_S=0, cnt=0. Condition holds.
                    #   So bad[c_idx] should increase by 1.
                    #   Let's verify:
                    #   Before: line had 0 S points. c_idx not in S.
                    #   For c_idx, punctured line had 0 S points. bad_contrib = 0.
                    #   After: line has 1 S point (c_idx). c_idx in S.
                    #   For c_idx, punctured line has 0 S points (only itself). bad_contrib = 0.
                    #   So bad[c_idx] should NOT increase?
                    #   Let's re-read the logic.
                    #   bad[x] = count of directions where punctured line intersects S.
                    #   Direction L.
                    #   Before: S ∩ L = {}. Punctured line for any x in L is L \ {x}.
                    #   S ∩ (L \ {x}) = {}. So bad_contrib = 0.
                    #   After: S ∩ L = {c}.
                    #   If x = c: Punctured line L \ {c}. S ∩ (L \ {c}) = {}. bad_contrib = 0.
                    #   If x != c: Punctured line L \ {x}. S ∩ (L \ {x}) = {c}. bad_contrib = 1.
                    #   So for x != c, bad increases by 1. For x = c, bad stays 0.
                    #   My previous logic said: increment x where OLD in_S[x]==0.
                    #   This includes c. So I would increment bad[c]. That is WRONG.
                    #   
                    #   Correction:
                    #   The condition for incrementing bad[x] is:
                    #   (OLD cnt > (1 if OLD in_S[x] else 0)) == False AND (NEW cnt > (1 if NEW in_S[x] else 0)) == True.
                    #   
                    #   Let's re-evaluate for cnt=0 (OLD).
                    #   NEW cnt = 1.
                    #   For x:
                    #     If OLD in_S[x]=0 (x not in S, x != c):
                    #       OLD: 0 > 0 is False.
                    #       NEW: 1 > 0 is True.
                    #       Change: +1.
                    #     If OLD in_S[x]=1 (x in S): Impossible since cnt=0 means no S points.
                    #     If x = c (OLD in_S=0):
                    #       OLD: 0 > 0 is False.
                    #       NEW in_S[c]=1.
                    #       NEW: 1 > 1 is False.
                    #       Change: 0.
                    
                    # So we increment bad[x] for all x in line where x != c_idx and not in_S[x] (current).
                    for x in line:
                        if x != c_idx and not in_S[x]:
                            bad[x] += 1
                
                elif cnt == 1:
                    # OLD cnt = 1. NEW cnt = 2.
                    # Let s be the unique point in S on L (OLD).
                    # For x:
                    #   If OLD in_S[x]=1 (x=s):
                    #     OLD: 1 > 1 is False.
                    #     NEW in_S[s]=1.
                    #     NEW: 2 > 1 is True.
                    #     Change: +1.
                    #   If OLD in_S[x]=0 (x != s, x != c):
                    #     OLD: 1 > 0 is True.
                    #     NEW: 2 > 0 is True.
                    #     Change: 0.
                    #   If x = c (OLD in_S=0):
                    #     OLD: 1 > 0 is True.
                    #     NEW in_S[c]=1.
                    #     NEW: 2 > 1 is True.
                    #     Change: 0.
                    
                    # So we increment bad[x] for x = s (the existing S point on the line).
                    # How to find s? It's the unique x in line with in_S[x]==1 (current, since c was not in S).
                    for x in line:
                        if in_S[x] and x != c_idx: # x must be the old S point
                            bad[x] += 1
                            break # Only one such point
                    
            # Note: We don't update bad[c_idx] in these cases?
            # Case cnt=0: bad[c] unchanged (0->0).
            # Case cnt=1: bad[c] unchanged (1->1).
            # So bad[c_idx] is not updated. This is correct.
            
            # What about other lines containing c?
            # We iterate over all lines containing c.
            # For each, we update line_s_counts and bad for relevant points.
            
            # Is it possible that c is on multiple lines? Yes, 50 lines.
            # We process each line independently.
            
            # Optimization: If time is tight, we can skip some checks?
            # But the check is O(49) per line, 50 lines -> 2450 ops per candidate.
            # 2401 candidates -> ~6M ops. Should be fast in Python?
            # 6M simple ops is < 1s.
            # The bottleneck is the Python loop overhead.
            # Let's hope it fits in 600s.
            
            # If we have time, we can try to add more?
            # The greedy is one pass. We could do multiple passes.
            # But one pass is usually enough for this kind of problem.
            
            # Let's add a time check inside the loop to be safe.
            if time.time() > _DEADLINE - 20:
                break

    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
