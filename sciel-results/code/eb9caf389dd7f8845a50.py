import numpy as np
import random
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3 in F_49

    # Field arithmetic helpers for F_49 = F_7[alpha] / (alpha^2 - 3)
    # Represented as (a, b) for a + b*alpha
    
    def add(a, b, c, d):
        return ((a + c) % p, (b + d) % p)
    
    def sub(a, b, c, d):
        return ((a - c) % p, (b - d) % p)
    
    def mul(a, b, c, d):
        # (a + b*alpha) * (c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = (ac + 3*bd) + (ad + bc)*alpha
        real = (a * c + k * b * d) % p
        imag = (a * d + b * c) % p
        return (real, imag)
    
    def inv(a, b):
        # Inverse of (a + b*alpha)
        # (a + b*alpha)(c + d*alpha) = 1
        # ac + 3bd = 1, ad + bc = 0
        # From second: d = -bc/a (if a != 0)
        # If a == 0, b != 0. Then 3bd = 1 => d = inv(3b). ad+bc = 0 => bc = 0 => c=0.
        # Norm = a^2 - 3b^2
        norm = (a * a - k * b * b) % p
        if norm == 0:
            return (0, 0)  # Should not happen for non-zero element in field
        norm_inv = pow(norm, -1, p)
        # (a - b*alpha) / norm
        real = (a * norm_inv) % p
        imag = (-b * norm_inv) % p
        return (real, imag)
    
    def div(a, b, c, d):
        return mul(a, b, *inv(c, d))
    
    def is_zero(a, b):
        return a == 0 and b == 0

    # Generate all points in F_49^2
    # Point is ((xa, xb), (ya, yb))
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    # We want to construct a small Nikodym set N.
    # A known construction for Nikodym sets in F_q^2 (q odd) is based on a conic.
    # Specifically, let C be a conic not containing the point at infinity in the vertical direction?
    # Actually, a simpler known result:
    # Let N be the set of points (x, y) such that y = f(x) where f is a function? No, that's a graph.
    # A graph of a function is not a Nikodym set because a horizontal line intersects it once, but we need the *other* points to be in N.
    # Wait, the definition: For every point x in F_q^2, there is a line L through x such that L \ {x} subset N.
    # If N is a graph of a function f, then for any x = (x0, y0), if y0 != f(x0), we need a line through x0 whose other points are in N.
    # A line with slope m through x0 is {(x0 + t, y0 + mt) | t in F_q}.
    # For this to be in N, we need y0 + mt = f(x0 + t) for all t != 0.
    # This implies f is a polynomial of degree at most 1? No, if f is quadratic, a line intersects it in at most 2 points.
    # So a graph cannot be a Nikodym set unless q is very small.
    
    # Correct construction:
    # Let N be the complement of a set S.
    # We need for every x in S, a line L_x through x such that L_x \ {x} subset N.
    # This means L_x \ {x} is disjoint from S.
    # So S is a set such that for every x in S, there is a line through x that intersects S only at x.
    # Such a set S is called a "blocking set" complement? No.
    # This is equivalent to S being a set of points where each point is "isolated" by some line.
    
    # Known small Nikodym sets:
    # For q=49, a construction by Ball and Blokhuis gives size q + sqrt(q) = 56.
    # The construction uses a conic.
    # Let C be a conic in F_q^2. Let N be the union of all lines tangent to C? No.
    # Let's try a specific algebraic construction.
    # Consider the set N of points (x, y) such that y^2 - x^3 - ax - b = 0? Elliptic curve.
    # Size approx q + 1.
    # If N is an elliptic curve, size ~ 50.
    # Does it satisfy the condition?
    # For any point x, is there a line through x such that the other points are on the curve?
    # A line intersects an elliptic curve in at most 3 points.
    # So L \ {x} has q-1 = 48 points. 48 > 3.
    # So N cannot be an elliptic curve.
    
    # The condition requires L \ {x} subset N.
    # So N must contain large portions of lines.
    # Specifically, if x in N, the line through x must be *mostly* in N.
    # If x in S (complement), the line through x must be *mostly* in N (disjoint from S except x).
    
    # Let's look at the current valid solution of size 147.
    # It uses a "cylinder" over a conic.
    # Let's try to improve the search for a smaller set.
    
    # Strategy:
    # 1. Start with a candidate set N of size k.
    # 2. Check validity.
    # 3. If valid, return.
    # 4. If not, try to reduce N.
    
    # A better approach: Construct S (the complement) directly.
    # We need S to be a set where every point x in S has a line through x intersecting S only at x.
    # This means S is a set of "isolated" points with respect to some line directions.
    
    # Let's try S to be a graph of a function with high degree?
    # No, if S is a graph, a line intersects it in few points.
    # If S is a graph of a function f, then for x=(x0, f(x0)) in S, we need a line through x0 such that no other point on the line is in S.
    # This is always true if the line is not tangent to the graph?
    # Wait. If L intersects S only at x, then L \ {x} is disjoint from S.
    # So L \ {x} subset N.
    # This is exactly the condition!
    # So if S is a set such that for every x in S, there exists a line through x that does not intersect S at any other point, then N = F_q^2 \ S is a Nikodym set.
    
    # When does a line through x not intersect S at any other point?
    # If S is a graph of a function f, a line with slope m through (x0, f(x0)) is y - f(x0) = m(x - x0).
    # Intersections with S: f(x) = f(x0) + m(x - x0).
    # We need this equation to have only one solution x = x0.
    # This is true if the function g(x) = f(x) - m(x - x0) - f(x0) has only one zero.
    # If f is a polynomial of degree d, this is a polynomial equation of degree d.
    # It has at most d roots.
    # We need it to have exactly 1 root (x0).
    # This is possible if we choose m such that x0 is a simple root and there are no other roots.
    # For a generic m, the number of roots is roughly d.
    # We need d=1? If f is linear, S is a line.
    # If S is a line, then for x in S, any line through x intersects S at x and potentially other points.
    # If we choose a line L through x that is not S, it intersects S only at x.
    # So if S is a line, for every x in S, we can choose L perpendicular to S (or any other slope).
    # Then L \ {x} is disjoint from S.
    # So N = F_q^2 \ S is a Nikodym set!
    # Size of N = q^2 - q = 49 * 48 = 2352.
    # This is huge. We want N small, so S large.
    
    # So we want S to be large, but each point in S must be "isolated" by some line.
    # If S is the entire plane, impossible.
    # If S is a union of lines?
    # Let S be a set of lines.
    # For x in S, we need a line L through x such that L \ {x} is disjoint from S.
    # If S is a single line, this works.
    # If S is two lines L1 and L2.
    # For x in L1 \ L2, we can choose L = L1? No, L1 \ {x} contains other points of L1 which are in S.
    # We need L \ {x} disjoint from S.
    # So L must not contain any other points of S.
    # If we choose L to be a line through x that is not L1 or L2, it intersects L1 at x and L2 at some other point y.
    # If y != x, then y is in S and y is in L. So L \ {x} contains y.
    # So L \ {x} is not disjoint from S.
    # Thus, S cannot be two lines.
    
    # What if S is a "grid" or something sparse?
    # We want S to be as large as possible.
    # The condition is: For every x in S, there is a line L_x through x such that L_x \cap S = {x}.
    # This means S is a set of points where each point has a "private" line direction.
    
    # Consider S to be a graph of a function f: F_q -> F_q.
    # |S| = q.
    # For x = (x0, f(x0)), we need a slope m such that the line y - f(x0) = m(x - x0) has no other intersections with the graph.
    # This requires the equation f(x) - f(x0) = m(x - x0) to have only solution x=x0.
    # If f is a permutation polynomial of high degree, this is likely for generic m.
    # But we need this to hold for *all* x0 in S.
    # If f is a polynomial of degree d, for a fixed x0, the equation has d roots.
    # We need to choose m such that x0 is the *only* root.
    # This is possible if we pick m from a large field.
    # But we need one m for each x0? No, the line L depends on x.
    # So for each x0, we can pick a different m0.
    # The condition is: There exists m0 such that x0 is the unique solution to f(x) - f(x0) = m0(x - x0).
    # This is equivalent to: The secant slope m0 = (f(x) - f(x0))/(x - x0) is unique to x=x0? No.
    # It means that for the chosen m0, the line with slope m0 through (x0, f(x0)) does not pass through any other (x, f(x)).
    # This is always true for a "generic" m0, unless the graph has collinear points.
    # If the graph has no 3 collinear points, then for any x0, and any other x1, the slope m(x0, x1) is unique to the pair.
    # If we pick m0 not equal to any m(x0, x1) for x1 != x0, then the line intersects the graph only at x0.
    # Since there are q-1 other points, there are q-1 forbidden slopes.
    # There are q slopes total.
    # So there is at least 1 allowed slope.
    # Thus, if S is a graph with no 3 collinear points, it satisfies the condition!
    
    # So we need a graph of a function f: F_q -> F_q with no 3 collinear points.
    # Such functions are called "3-collinear free" or similar.
    # A random function has many 3-collinear points.
    # A polynomial of degree 2 (parabola) has no 3 collinear points?
    # Line intersects parabola in at most 2 points. Yes.
    # So f(x) = x^2 works.
    # S = {(x, x^2) | x in F_q}.
    # |S| = q = 49.
    # |N| = q^2 - q = 2352.
    # This is the same as the line case. Not good.
    
    # We need |S| to be large.
    # Can we take a union of graphs?
    # Let S be a union of k graphs.
    # |S| = k * q.
    # For x in S, it belongs to one graph Gi.
    # We need a line L through x such that L \ {x} is disjoint from S.
    # This means L cannot intersect any graph Gj at any point other than x.
    # If x is in Gi, L intersects Gi at x and potentially other points.
    # We need L to intersect Gi only at x, and Gj (j!=i) at no points.
    # This is harder.
    
    # However, if we take S to be a "large" set with the property that each point has a private line, we win.
    # What is the maximum size of such a set S?
    # This is related to the size of a "Kakeya set" complement?
    # Actually, the Nikodym set N is the complement of S.
    # We want N small, so S large.
    # The minimum size of a Nikodym set in F_q^2 is known to be at least q + sqrt(q).
    # So |N| >= 56.
    # This implies |S| <= 2401 - 56 = 2345.
    
    # The construction achieving q + sqrt(q) uses a conic.
    # Let C be a conic. Let T be the set of tangents to C.
    # Let N be the union of all points on the tangents?
    # No, the standard construction is:
    # Let N be the set of points (x, y) such that y = x^2 + a x + b for some a, b?
    # Let's look up "Nikodym set finite field construction".
    # One construction: Let S be the set of points (x, y) such that y = f(x) where f is a permutation polynomial?
    # No, that gives |S|=q.
    
    # Wait, the question asks for N.
    # Current best N has size 147.
    # Theoretical lower bound is 56.
    # Let's try to construct N of size 56.
    
    # Construction from literature (e.g., Ball and Blokhuis):
    # Let q = p^2.
    # Let C be a conic in F_q^2.
    # Let N be the set of points on the lines tangent to C?
    # There are q+1 tangents. Each has q points.
    # Union size is roughly q^2. Too big.
    
    # Another construction:
    # N is the set of points (x, y) such that y^2 - x^3 - ax - b = 0? No.
    
    # Let's try a different approach: Local Search.
    # Start with a valid set N (size 147).
    # Try to remove points one by one.
    # If removing a point x breaks validity, keep it.
    # If not, remove it.
    
    # Validity check:
    # For every point p in F_q^2, there is a line L through p such that L \ {p} subset N.
    
    # If we remove x from N, we need to ensure that for every p, the condition still holds.
    # The only p for which the condition might fail are those that relied on x being in N.
    # Specifically, if there was a line L through p such that L \ {p} subset N_old, and x in L \ {p}, then L \ {p} is no longer subset N_new.
    # But p might have another line L' such that L' \ {p} subset N_new.
    
    # So, to check if we can remove x:
    # For every p such that x is in some valid line for p, check if p has another valid line in N_new.
    
    # This is expensive.
    # Let's implement a greedy removal.
    
    # 1. Build N_old (size 147).
    # 2. For each x in N_old (in random order):
    #    a. Tentatively remove x.
    #    b. Check if N_new is valid.
    #    c. If valid, keep x removed. Else, restore x.
    
    # How to check validity efficiently?
    # Precompute for each point p, the set of lines through p that are "fully contained" in N (except p).
    # A line is defined by (p, m) where m is slope or vertical.
    # There are q+1 lines through each p.
    # Total lines = q^2 * (q+1) / 2? No, each line has q points.
    # Number of lines in F_q^2 is q(q+1).
    # For each line L, we can check if L \ {p} subset N for each p in L.
    # This is O(q^2 * q) = O(q^3) = 117,649.
    # This is fast!
    
    # Algorithm:
    # 1. Initialize N to the 147-point set.
    # 2. Compute for each point p, the number of valid lines through p (lines L through p such that L \ {p} subset N).
    #    Let count[p] be this number.
    #    N is valid iff count[p] > 0 for all p.
    # 3. To remove x:
    #    a. For each line L containing x, update counts for all p in L.
    #       If L was valid for p (i.e., L \ {p} subset N), and x in L \ {p}, then L is no longer valid for p in N_new.
    #       So count[p] decreases by 1 if L was the only valid line? No, just decrement.
    #    b. Check if any count[p] becomes 0.
    #    c. If yes, restore x.
    #    d. If no, keep x removed.
    
    # Complexity:
    # Each removal involves updating counts for q+1 lines, each with q points.
    # So O(q^2) per removal.
    # Total O(q^3) for all removals.
    # q=49, q^3 = 117,649. Very fast.
    
    # Let's implement this.
    
    # First, generate the 147-point set.
    # The current code uses:
    # in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
    # in_y_special = (ya, yb) in y_special where y_special = {(u^2, v)}
    # N = { (xa, xb, ya, yb) | in_unital and not in_y_special }
    
    # Let's verify the size and validity of this set first.
    
    def generate_baseline_n():
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add((u * u % p, v))
                
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            points.append(((xa, xb), (ya, yb)))
        return set(points)
    
    N = generate_baseline_n()
    
    # Map points to indices
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    idx_to_point = {i: pt for i, pt in enumerate(all_points)}
    
    # Precompute lines
    # A line is defined by a point and a slope.
    # Slopes: 0..p-1 for non-vertical, p for vertical.
    # Actually, slopes in F_49.
    # There are q slopes + 1 vertical = q+1 directions.
    
    # List of all lines: list of sets of points.
    lines = []
    # For each direction d in F_49 U {inf}
    # For each intercept...
    # Easier: For each point p and each direction d, generate the line.
    # But many duplicates.
    # Let's just iterate over all points and all directions, and store the line as a frozenset of points.
    
    line_set = set()
    for p_idx, p in enumerate(all_points):
        for d in range(q + 1):
            if d == q:
                # Vertical line: x = p.x
                pts = [pt for pt in all_points if pt[0] == p[0]]
            else:
                # Slope d = (da, db)
                da, db = divmod(d, p)
                # y - p.y = d * (x - p.x)
                pts = []
                for t in range(q):
                    # t = (ta, tb)
                    ta, tb = divmod(t, p)
                    # x = p.x + t
                    # y = p.y + d * t
                    px, py = p
                    tx, ty = add(px[0], px[1], ta, tb)
                    dy = mul(da, db, ta, tb)
                    ty = add(py[0], py[1], dy[0], dy[1])
                    pts.append(((tx[0], tx[1]), (ty[0], ty[1])))
            line_tuple = tuple(sorted(pts, key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1])))
            if line_tuple not in line_set:
                line_set.add(line_tuple)
    
    lines = list(line_set)
    
    # For each line, list of point indices
    line_points = []
    for line in lines:
        pts_idx = [point_to_idx[pt] for pt in line]
        line_points.append(pts_idx)
    
    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q * q)]
    for l_idx, pts_idx in enumerate(line_points):
        for pt_idx in pts_idx:
            point_lines[pt_idx].append(l_idx)
    
    # counts[p] = number of lines L through p such that L \ {p} subset N
    # Initially, compute counts for N.
    counts = [0] * (q * q)
    
    # For each line, check if it is "valid" for each point on it.
    # A line L is valid for p if all other points in L are in N.
    # Let's compute for each line L, the number of points in L \ N.
    # If count_not_in_N == 0, then L \ {p} subset N for all p in L.
    # If count_not_in_N == 1, let x be the point not in N.
    # Then L \ {p} subset N for all p in L except p=x.
    # For p=x, L \ {x} contains all other points which are in N. So L is valid for x.
    # Wait.
    # Condition: L \ {p} subset N.
    # If x not in N, and x in L.
    # If p = x, L \ {x} does not contain x. So if all other points are in N, it's valid.
    # If p != x, L \ {p} contains x. So it's not valid.
    
    # So:
    # For each line L:
    #   not_in_N = [pt for pt in L if pt not in N]
    #   if len(not_in_N) == 0:
    #       for p in L: counts[p] += 1
    #   elif len(not_in_N) == 1:
    #       x = not_in_N[0]
    #       counts[x] += 1  # L is valid for x
    #   else:
    #       # No point p has L \ {p} subset N
    #       pass
    
    for pts_idx in line_points:
        not_in_N = [pt for pt in pts_idx if idx_to_point[pt] not in N]
        if len(not_in_N) == 0:
            for pt in pts_idx:
                counts[pt] += 1
        elif len(not_in_N) == 1:
            x = not_in_N[0]
            counts[x] += 1
    
    # Check if N is valid
    if any(c == 0 for c in counts):
        print("Baseline N is not valid!")
        # Return baseline anyway
        return np.array([list(idx_to_point[i]) for i in range(q*q) if idx_to_point[i] in N], dtype=np.int64)
    
    # Greedy removal
    # Order points in N randomly
    n_points = list(N)
    random.seed(42)
    random.shuffle(n_points)
    
    for pt in n_points:
        pt_idx = point_to_idx[pt]
        # Tentatively remove pt
        N.remove(pt)
        
        # Update counts
        # For each line L containing pt:
        #   Previously, L had some number of points not in N.
        #   Now pt is not in N.
        #   We need to re-evaluate validity for all points in L.
        #   This is complex to update incrementally.
        #   Given q=49, let's just recompute counts from scratch.
        #   O(q^3) = 117k operations.
        #   We do this for each removal attempt.
        #   Up to 147 removals.
        #   Total 147 * 117k = 17M operations.
        #   This might be slow in Python.
        #   Let's optimize.
        
        # Recompute counts
        new_counts = [0] * (q * q)
        for pts_idx in line_points:
            not_in_N = [pt for pt in pts_idx if idx_to_point[pt] not in N]
            if len(not_in_N) == 0:
                for pt in pts_idx:
                    new_counts[pt] += 1
            elif len(not_in_N) == 1:
                x = not_in_N[0]
                new_counts[x] += 1
        
        if any(c == 0 for c in new_counts):
            # Invalid, restore
            N.add(pt)
        else:
            # Valid, keep removed
            counts = new_counts
            
    # Return N
    result = []
    for pt in N:
        result.append([list(pt[0]), list(pt[1])])
    
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
