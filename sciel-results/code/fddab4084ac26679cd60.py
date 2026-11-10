import math
import time
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Determinant | x1 y1 1 |
                #             | x2 y2 1 |
                #             | x3 y3 1 |
                det = x1 * (y2 - y3) - y1 * (x2 - x3) + (x2 * y3 - x3 * y2)
                if det == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # Determinant | x^2+y^2 x y 1 |
                    #             | x^2+y^2 x y 1 |
                    #             | x^2+y^2 x y 1 |
                    #             | x^2+y^2 x y 1 |
                    d1 = x1*x1 + y1*y1
                    d2 = x2*x2 + y2*y2
                    d3 = x3*x3 + y3*y3
                    d4 = x4*x4 + y4*y4
                    
                    # We can check if the 4 points are concyclic by checking if the circumcircle of 1,2,3 passes through 4.
                    # Or use the determinant.
                    # To avoid large numbers, we can use the property that 4 points are concyclic iff the determinant is 0.
                    # But calculating 4x4 det with large ints is fine.
                    
                    # Let's use the determinant formula for concyclicity:
                    # | x1 y1 x1^2+y1^2 1 |
                    # | x2 y2 x2^2+y2^2 1 |
                    # | x3 y3 x3^2+y3^2 1 |
                    # | x4 y4 x4^2+y4^2 1 | = 0
                    
                    # We can subtract row 1 from others to simplify?
                    # Or just compute.
                    
                    # A faster check:
                    # Calculate circumcenter of 1,2,3. If it's undefined (collinear), we already failed.
                    # Check if dist(center, 1) == dist(center, 4).
                    # This involves fractions.
                    
                    # Let's stick to the determinant.
                    # To speed up, we can check if the power of point 4 with respect to circle 1-2-3 is 0.
                    # Power = x4^2 + y4^2 + Dx4 + Ey4 + F
                    # where D, E, F are derived from 1,2,3.
                    
                    # Equation of circle: x^2 + y^2 + Dx + Ey + F = 0
                    # x1^2 + y1^2 + D x1 + E y1 + F = 0
                    # x2^2 + y2^2 + D x2 + E y2 + F = 0
                    # x3^2 + y3^2 + D x3 + E y3 + F = 0
                    
                    # Let A = x1^2+y1^2, B = x2^2+y2^2, C = x3^2+y3^2
                    # A + D x1 + E y1 + F = 0
                    # B + D x2 + E y2 + F = 0
                    # C + D x3 + E y3 + F = 0
                    
                    # Subtract 1st from 2nd: B-A + D(x2-x1) + E(y2-y1) = 0
                    # Subtract 1st from 3rd: C-A + D(x3-x1) + E(y3-y1) = 0
                    
                    # Let u = x2-x1, v = y2-y1
                    # Let w = x3-x1, z = y3-y1
                    # D u + E v = A-B
                    # D w + E z = A-C
                    
                    # Det = u*z - v*w
                    # If Det == 0, collinear (handled).
                    # D = ( (A-B)z - (A-C)v ) / Det
                    # E = ( u(A-C) - w(A-B) ) / Det
                    
                    # F = -A - D x1 - E y1
                    
                    # Check point 4:
                    # x4^2 + y4^2 + D x4 + E y4 + F == 0
                    # Multiply by Det to avoid fractions:
                    # (x4^2+y4^2)Det + D_num x4 Det + E_num y4 Det + F_num Det == 0 ?
                    # Actually, just substitute D_num/Den into the equation.
                    # Let D_num = (A-B)z - (A-C)v
                    # Let E_num = u(A-C) - w(A-B)
                    # Let F_num = -A*Det - D_num*x1 - E_num*y1
                    
                    # Check: (x4^2+y4^2)*Det + D_num*x4 + E_num*y4 + F_num == 0
                    
                    A = x1*x1 + y1*y1
                    B = x2*x2 + y2*y2
                    C = x3*x3 + y3*y3
                    D4 = x4*x4 + y4*y4
                    
                    u = x2 - x1
                    v = y2 - y1
                    w = x3 - x1
                    z = y3 - y1
                    
                    Det = u*z - v*w
                    if Det == 0:
                        continue # Collinear, already checked, but safe
                    
                    D_num = (A-B)*z - (A-C)*v
                    E_num = u*(A-C) - w*(A-B)
                    F_num = -A*Det - D_num*x1 - E_num*y1
                    
                    val = D4*Det + D_num*x4 + E_num*y4 + F_num
                    if val == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known 5-point set
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = fallback_points
    best_size = 5
    
    # Search for larger cliques in integer lattice
    # We search for a clique of size 6 first.
    # If found, return it.
    
    def find_clique_size_k(points, k):
        # points is a list of tuples
        # Returns a list of k points if found, else None
        # Greedy approach with backtracking for small k
        # We build adjacency sets
        n = len(points)
        adj = [set() for _ in range(n)]
        
        # Precompute squared distances
        # This is O(N^2)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                d2 = dx*dx + dy*dy
                if _is_square(d2):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find clique of size k
        # Use a recursive search
        def backtrack(candidates, current_clique):
            if len(current_clique) == k:
                return current_clique[:]
            if not candidates:
                return None
            
            # Heuristic: pick vertex with highest degree in subgraph induced by candidates
            # But for speed, just pick first or random?
            # Let's pick the one with most neighbors in candidates
            best_v = -1
            best_deg = -1
            for v in candidates:
                deg = len(adj[v] & set(candidates))
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            new_clique = current_clique + [v]
            new_candidates = [u for u in candidates if u != v and u in adj[v]]
            
            res = backtrack(new_candidates, new_clique)
            if res:
                return res
            return None
        
        # Initial candidates: all vertices
        # To speed up, we can filter vertices with degree < k-1
        initial_candidates = [i for i in range(n) if len(adj[i]) >= k-1]
        return backtrack(initial_candidates, [])

    # Try radii
    radii = [50, 100, 200, 500]
    for R in radii:
        if time.time() > deadline - 10:
            break
            
        # Generate points
        pts = []
        for x in range(-R, R+1):
            for y in range(-R, R+1):
                pts.append((x, y))
        
        # Find clique of size 6
        clique = find_clique_size_k(pts, 6)
        if clique:
            # Verify geometric constraints
            if _check_no_three_collinear(clique) and _check_no_four_concyclic(clique):
                return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in clique]}
        
        # If 6 not found, try 5? We already have 5.
        # But maybe a different 5 is better? No, score is likely just count.
        # The task says "Maximize the number of points".
        # So 6 is a jump from 5.
        
        # If we don't find 6, we can try to find a 6 in a smaller set of points?
        # The search above is O(N^2) for graph construction and potentially exponential for clique.
        # For R=50, N=10000. Graph construction is fast.
        # Clique search might be slow.
        
    # If no 6 found, return the known 5
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback_points]}
