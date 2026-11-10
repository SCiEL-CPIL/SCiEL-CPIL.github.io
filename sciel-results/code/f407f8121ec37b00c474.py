# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Start with the known valid baseline construction (Hermitian unital complement).
    2. Use an efficient greedy algorithm to iteratively remove points from N,
       checking validity via precomputed "clear counts" for each point's witness lines.
    
    The key efficiency gain is maintaining `clear_counts` which tracks how many
    points on each potential witness line are NOT in N. When removing a point,
    we update these counts in O(1) per affected line. Validity check is O(1).
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, smallest quadratic non-residue mod 7
    
    # F_49 arithmetic: (a, b) represents a + b*alpha, alpha^2 = 3
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb), indexed 0..2400
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute all lines (vertical and slope lines)
    # Each line is a set of point indices
    # For each point, store which lines pass through it (potential witness lines)
    
    # Vertical lines: fixed (xa, xb), all (ya, yb)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    
    # Slope lines: direction (1, m) where m = (ma, mb) != (0,0)
    # Line: {x + t*(1, m) : t in F_49}
    # For point (x, y), the line through it with direction (1,m) has "intercept" 
    # y - m*x (in the sense of the line equation)
    
    slope_lines = {}  # (ma, mb) -> {intercept_key: [point_indices]}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            slope_lines[(ma, mb)] = dict(lines)
    
    # For each point, precompute its potential witness lines
    # A witness line for point i is a line L containing i such that L\{i} subset N
    # We need to track for each point, for each line through it, how many points on that line are NOT in N
    # (excluding the point itself)
    
    # in_N[i] = True if point i is in the Nikodym set
    in_N = [True] * n
    
    # Initialize with baseline construction
    # Exclude points on Hermitian unital U where y_a is a square mod p
    # U: {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*3
    # So condition: 2*xa + ya^2 - 3*yb^2 ≡ 0 (mod 7)
    # y_a must be a quadratic residue mod 7: {0, 1, 2, 4}
    qr = set((u*u) % p for u in range(p))  # {0, 1, 2, 4}
    
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
        if in_unital and ya in qr:
            in_N[i] = False
    
    # Now compute clear_counts for each point
    # clear_counts[i] = number of potential witness lines for point i where ALL other points on the line are in N
    # Actually, we need to be more careful. Let's define:
    # For point i, a line L through i is a "good witness" if all points in L\{i} are in N.
    # We want at least one good witness for each point.
    
    # Let's define for each point i and each line L through i:
    #   not_in_N_on_L = number of points j in L, j != i, such that not in_N[j]
    #   L is a good witness for i iff not_in_N_on_L == 0
    
    # We'll maintain for each point i, the count of good witness lines.
    # good_witness_count[i] = number of lines L through i where not_in_N_on_L == 0
    
    # Precompute for each point, the list of lines through it
    # Line representation: ('vert', (xa, xb)) or ('slope', (ma, mb), key)
    
    point_lines = [[] for _ in range(n)]
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        # Vertical line
        point_lines[i].append(('vert', (xa, xb), None))
        # Slope lines
        for ma in range(p):
            for mb in range(p):
                if ma == 0 and mb == 0:
                    continue
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                point_lines[i].append(('slope', (ma, mb), key))
    
    # For each line, compute not_in_N count (excluding the point we're considering)
    # This is tricky because not_in_N depends on which point we're considering.
    # Instead, let's define:
    # For a line L, let total_not_in_N = number of points j in L with not in_N[j]
    # For point i in L, not_in_N_on_L_excluding_i = total_not_in_N - (1 if not in_N[i] else 0)
    # L is a good witness for i iff not_in_N_on_L_excluding_i == 0
    
    # So we need for each line L: total_not_in_N[L]
    # And for each point i: good_witness_count[i] = number of lines L through i where 
    #   total_not_in_N[L] - (1 if not in_N[i] else 0) == 0
    
    # Let's maintain:
    # line_not_in_N_count[L] = number of points in L with not in_N[j]
    # point_good_witness[i] = number of lines L through i where line_not_in_N_count[L] - (1 if not in_N[i] else 0) == 0
    
    # Initialize line_not_in_N_count
    line_not_in_N_count = {}
    # Vertical lines
    for key, pts in vert_lines.items():
        cnt = sum(1 for j in pts if not in_N[j])
        line_not_in_N_count[('vert', key)] = cnt
    
    # Slope lines
    for (ma, mb), lines in slope_lines.items():
        for key, pts in lines.items():
            cnt = sum(1 for j in pts if not in_N[j])
            line_not_in_N_count[('slope', (ma, mb), key)] = cnt
    
    # Initialize point_good_witness
    point_good_witness = [0] * n
    for i in range(n):
        cnt = 0
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                key = line_rep[1]
                lkey = ('vert', key)
            else:
                ma, mb = line_rep[1]
                key = line_rep[2]
                lkey = ('slope', (ma, mb), key)
            
            total_nin = line_not_in_N_count[lkey]
            nin_excl_i = total_nin - (0 if in_N[i] else 1)
            if nin_excl_i == 0:
                cnt += 1
        point_good_witness[i] = cnt
    
    # Verify initial validity: all points should have point_good_witness[i] > 0
    # (This should be true for the baseline construction)
    
    # Now greedily try to remove points from N
    # A point i can be removed from N if:
    # 1. Currently in_N[i] == True
    # 2. After removal, all points still have at least one good witness
    #    - For point i itself: it needs a good witness line L where all other points on L are in N.
    #      After removing i, i is not in N, so for i to have a good witness, we need a line L through i
    #      where all j in L\{i} are in N. This is the same as before: line_not_in_N_count[L] == 0
    #      (since i was in N, removing it doesn't change line_not_in_N_count for lines through i... 
    #       wait, no. line_not_in_N_count counts points NOT in N. If i was in N and we remove it,
    #       line_not_in_N_count increases by 1 for all lines through i.)
    #    
    #    Let's reconsider. When we remove point i from N:
    #    - For each line L through i: line_not_in_N_count[L] increases by 1
    #    - For each point j in L (j != i): 
    #      The status of L as a good witness for j may change.
    #      Before: L good for j iff line_not_in_N_count[L] - (1 if not in_N[j] else 0) == 0
    #      After: line_not_in_N_count[L] increases by 1, so:
    #      L good for j iff (line_not_in_N_count[L] + 1) - (1 if not in_N[j] else 0) == 0
    #      i.e., line_not_in_N_count[L] - (1 if not in_N[j] else 0) == -1
    #      Since the left side was >= 0 before, it can only become -1 if it was 0 before.
    #      So L stops being a good witness for j iff it was a good witness before.
    #    
    #    - For point i itself:
    #      Before: L good for i iff line_not_in_N_count[L] - (1 if not in_N[i] else 0) == 0
    #      Since in_N[i] was True, this is line_not_in_N_count[L] == 0.
    #      After: in_N[i] is False, so L good for i iff (line_not_in_N_count[L] + 1) - 1 == 0
    #      i.e., line_not_in_N_count[L] == 0. Same condition!
    #      So i's good witness count doesn't change when we remove i.
    
    #    So the algorithm is:
    #    - Try to remove point i (in_N[i] = False)
    #    - For each line L through i:
    #      - line_not_in_N_count[L] += 1
    #      - For each point j in L, j != i:
    #        - If L was a good witness for j (i.e., line_not_in_N_count[L] - (1 if not in_N[j] else 0) was 0),
    #          then it's no longer a good witness, so point_good_witness[j] -= 1
    #    - Check if any point j has point_good_witness[j] == 0. If so, revert.
    #    - If no point has point_good_witness[j] == 0, keep the removal.
    
    # To make this efficient, we need to quickly find which points j are affected.
    # The affected points are all j in any line L through i, j != i.
    # This is at most 49*48 = 2352 points (49 lines through i, each with 48 other points, but with overlaps).
    # Actually, each point j (j != i) is on exactly one vertical line with i and one slope line with i for each slope.
    # So the total number of distinct points j != i is 49*49 - 1 = 2400, but they appear in multiple lines.
    # The total number of (j, L) pairs where L is a line through i and j in L, j != i is 49*48 = 2352.
    
    # This is O(2352) per candidate, which is fine.
    
    # Let's implement the greedy removal
    
    # Candidate points to remove: points currently in N
    # We'll try in some order. Let's try all points, maybe multiple passes.
    
    # To be efficient, we'll keep a list of candidates and iterate.
    # We can also prioritize points that are "easy" to remove (e.g., those with many good witnesses).
    
    # For now, let's just iterate through all points in order, multiple passes if time allows.
    
    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() > deadline - 30:
            break
        
        made_progress = False
        for i in range(n):
            if time.time() > deadline - 30:
                break
            
            if not in_N[i]:
                continue  # Already removed
            
            # Try to remove point i
            # First, check if i has at least one good witness line
            if point_good_witness[i] == 0:
                continue  # Can't remove i, it has no good witness
            
            # Simulate removal
            affected = []  # List of (j, lkey, was_good) tuples
            lines_through_i = point_lines[i]
            
            # Check if after removal, i still has a good witness
            # i's good witness count doesn't change, so if it was > 0, it's still > 0.
            # But we need to check other points.
            
            # For each line L through i:
            for line_rep in lines_through_i:
                if line_rep[0] == 'vert':
                    key = line_rep[1]
                    lkey = ('vert', key)
                    line_pts = vert_lines[key]
                else:
                    ma, mb = line_rep[1]
                    key = line_rep[2]
                    lkey = ('slope', (ma, mb), key)
                    line_pts = slope_lines[(ma, mb)][key]
                
                # line_not_in_N_count[lkey] will increase by 1
                old_count = line_not_in_N_count[lkey]
                new_count = old_count + 1
                
                # For each j in line_pts, j != i:
                for j in line_pts:
                    if j == i:
                        continue
                    # Was L a good witness for j?
                    # Condition: old_count - (1 if not in_N[j] else 0) == 0
                    was_good = (old_count - (0 if in_N[j] else 1)) == 0
                    if was_good:
                        affected.append((j, lkey, True))
            
            # Check if any affected point j would have point_good_witness[j] == 0 after removal
            # We need to count how many times each j appears in affected (with was_good=True)
            # and subtract from point_good_witness[j]
            
            # For efficiency, let's use a dictionary to count
            decrement_counts = defaultdict(int)
            for j, lkey, was_good in affected:
                if was_good:
                    decrement_counts[j] += 1
            
            # Check if any j would have point_good_witness[j] - decrement_counts[j] < 0
            # Actually, it should be >= 0, but we check if it becomes 0
            invalid = False
            for j, dec in decrement_counts.items():
                if point_good_witness[j] - dec < 0:
                    invalid = True
                    break
                # If it becomes 0, that's okay as long as it's not negative
                # But we need at least 1 good witness, so if it becomes 0, that's a problem
                # Wait, point_good_witness[j] counts the number of good witness lines.
                # If it becomes 0, then j has no good witness, which is invalid.
                if point_good_witness[j] - dec == 0:
                    # This is a problem only if j != i, because i is being removed from N,
                    # but i still needs a good witness (which it has, as checked above)
                    # Actually, all points in F_49^2 need a good witness, including points not in N.
                    # So if any point j (j != i) has point_good_witness[j] - dec == 0, it's invalid.
                    invalid = True
                    break
            
            if invalid:
                continue  # Can't remove i
            
            # Remove i
            in_N[i] = False
            made_progress = True
            
            # Update data structures
            for line_rep in lines_through_i:
                if line_rep[0] == 'vert':
                    key = line_rep[1]
                    lkey = ('vert', key)
                else:
                    ma, mb = line_rep[1]
                    key = line_rep[2]
                    lkey = ('slope', (ma, mb), key)
                
                line_not_in_N_count[lkey] += 1
            
            for j, lkey, was_good in affected:
                if was_good:
                    point_good_witness[j] -= 1
        
        if not made_progress:
            break  # No more progress possible
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
