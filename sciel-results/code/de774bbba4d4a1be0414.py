# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy:
    1. Start with all 2401 points in N.
    2. For each point x, determine if it can be "excluded" (removed from N).
       A point x can be excluded if there exists a direction v such that
       all other points on the line through x with direction v are in N.
       However, this is circular. Instead, we use a greedy approach:
       
       We try to remove points from N one by one. A point x can be removed
       if, for at least one direction v, all other points on the line {x + t*v : t != 0}
       are currently in N. But wait, if we remove x, then for any OTHER point y on that line,
       the line through y in direction v would now have x missing, so y might not be coverable
       by that specific line. But y might be coverable by a DIFFERENT line.
       
       The condition is: For EVERY point x in F_q^2, there EXISTS a line L_x containing x
       such that L_x \ {x} subset N.
       
       This is equivalent to: The set of excluded points S = F_q^2 \ N is such that
       for every x, there is a line through x that does NOT contain any other point of S.
       
       In other words, no two points of S can lie on a line where one "blocks" the other?
       No, that's not quite right.
       
       Let's rephrase. Let S be the set of excluded points.
       Condition: For every x in F_q^2, there exists a line L through x such that L \ {x} ∩ S = ∅.
       This means that x is NOT "covered" by any line that contains another point of S?
       Actually, if a line L contains x and some s in S (s != x), then L is NOT a valid line for x.
       So, x is valid if there is at least one line through x that contains NO other point of S.
       
       Therefore, S must be a set such that for every x (including x in S), there is a line through x
       that is "S-free" except possibly at x.
       
       This is exactly the definition of a "scattered" set or something similar?
       Actually, this looks like we want S to be as large as possible such that
       no point x is "trapped" by S on all lines through it.
       
       Greedy approach:
       Start with S = ∅.
       Try to add points to S one by one.
       When adding a point s to S, we must ensure that for every x in F_q^2,
       there still exists a line through x that contains no other point of S.
       
       This check is expensive: O(|S| * 2401 * 50).
       
       Alternative view:
       A point x becomes "invalid" (i.e., cannot be part of the Nikodym set's complement? No, x is a point in the space)
       Wait, the Nikodym set N is the set of points that are NOT in S.
       The condition is on N.
       For every x in F_q^2, there is a line through x such that all other points on that line are in N.
       This is equivalent to: For every x, there is a line through x such that no other point on that line is in S.
       
       So, if we add a point s to S, it "blocks" all lines that pass through s.
       Specifically, for any other point x on a line with s, that line is no longer a valid "clear" line for x.
       x remains valid if there is at least one line through x that does NOT pass through any point in S (except possibly x itself if x is in S).
       
       Let's define a point x as "covered" by S if EVERY line through x contains some point of S \ {x}.
       We must ensure that no point x is covered by S.
       
       Greedy:
       Start with S = ∅.
       For each point s in some order:
           Tentatively add s to S.
           Check if any point x becomes covered.
           If yes, revert.
           If no, keep s.
       
       How to check if adding s to S causes any x to become covered?
       Adding s to S means that for any x on a line with s, the set of available "clear" lines for x shrinks.
       x becomes covered if the number of clear lines for x drops to 0.
       
       We can maintain a counter for each x: `clear_lines[x]` = number of lines through x that contain no points of S \ {x}.
       Initially, clear_lines[x] = 50 for all x.
       When we add s to S:
           For each of the 50 lines L through s:
               For each other point x on L:
                   clear_lines[x] -= 1
                   If clear_lines[x] becomes 0, then x is now covered. This is a conflict.
       
       So, adding s is valid if and only if no x on any line through s has clear_lines[x] == 1 (before the decrement).
       Because if clear_lines[x] was 1, and we decrement it to 0, x becomes covered.
       
       So the check is:
       For each of the 50 lines through s:
           For each other point x on that line:
               If clear_lines[x] == 1:
                   Conflict!
       
       If no conflict, we can add s.
       Then we update:
       For each of the 50 lines through s:
           For each other point x on that line:
               clear_lines[x] -= 1
       
       This is O(50 * 49) per candidate s.
       Total for all 2401 points: 2401 * 50 * 49 ≈ 5.9 million operations.
       This is very fast in Python if optimized, or with NumPy.
       
       We can try multiple random orders to see if we get a larger S.
       """
    p = 7
    q = p * p
    total_points = q * q
    
    # Precompute field operations
    # F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
        
    # Generate all points
    points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    point_index = {pt: i for i, pt in enumerate(points)}
    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, 1, mb)) # (1, m)
    directions.append((0, 1, 0, 1)) # (0, 1)
    
    # For each point, store the list of other points on each line through it
    # lines_through[i] = list of 50 lists, each list contains 49 point indices
    lines_through = []
    for i, pt in enumerate(points):
        pt_lines = []
        for d in directions:
            line_pts = []
            for t in range(q):
                if t == 0:
                    continue
                # x + t*v
                # v = (vx, vy), x = (x, y)
                # t is in F_49
                ta, tb = t // p, t % p
                t_val = (ta, tb)
                
                # t * v
                tvx = mul(t_val, (d[0], d[1]))
                tvy = mul(t_val, (d[2], d[3]))
                
                # x + t*v
                nx = add((pt[0], pt[1]), tvx)
                ny = add((pt[2], pt[3]), tvy)
                
                npt = (nx[0], nx[1], ny[0], ny[1])
                line_pts.append(point_index[npt])
            pt_lines.append(line_pts)
        lines_through.append(pt_lines)
        
    # Greedy search
    # clear_lines[i] = number of lines through point i that are clear of S (excluding i)
    
    def try_greedy(order):
        S = set()
        clear_lines = np.full(total_points, 50, dtype=np.int32)
        
        for s_idx in order:
            if s_idx in S:
                continue
            # Check if adding s_idx causes any conflict
            conflict = False
            for line_pts in lines_through[s_idx]:
                for x_idx in line_pts:
                    if clear_lines[x_idx] == 1:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                S.add(s_idx)
                for line_pts in lines_through[s_idx]:
                    for x_idx in line_pts:
                        clear_lines[x_idx] -= 1
        
        return S

    # Try a few random orders
    best_S = set()
    random.seed(42)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600
    
    # Try default order first
    order = list(range(total_points))
    S = try_greedy(order)
    if len(S) > len(best_S):
        best_S = S
    
    # Try random orders
    while time.time() < deadline - 10:
        order = list(range(total_points))
        random.shuffle(order)
        S = try_greedy(order)
        if len(S) > len(best_S):
            best_S = S
    
    # Construct N
    N = []
    for i, pt in enumerate(points):
        if i not in best_S:
            N.append([[pt[0], pt[1]], [pt[2], pt[3]]])
    
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
