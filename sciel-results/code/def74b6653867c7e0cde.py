# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Generate all points in F_49^2 as tuples (xa, xb, ya, yb)
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))
    
    # Identify points in the Hermitian Unital U
    # Equation: 2*xa + ya^2 - 3*yb^2 == 0 (mod 7)
    points_in_U = []
    for pt in all_points:
        xa, xb, ya, yb = pt
        if (2 * xa + ya * ya - 3 * yb * yb) % p == 0:
            points_in_U.append(pt)
            
    # Precompute lines for each point to speed up validity check?
    # Or just iterate on the fly.
    
    def get_lines_through(point):
        """
        Yields all lines passing through 'point' in F_49^2.
        A line is represented by (v_x, v_y) direction, but we need the set of points.
        Actually, for validity check, we just need to know if the line intersects S.
        We can represent a line by its direction (mx, my) and a point on it.
        But there are 50 lines through a point.
        Directions: (1, m) for m in F_49 (49 lines) and (0, 1) (1 line).
        F_49 elements are (a, b).
        m = (ma, mb).
        """
        # We will generate the set of points on the line for each direction
        # and check intersection with S.
        pass

    def is_valid_complement(S_set):
        """
        S_set is a set of tuples (xa, xb, ya, yb).
        Checks if N = All \ S is a Nikodym set.
        Condition: For every x in All, there exists a line L through x
        such that (L \ {x}) is subset of N.
        This is equivalent to:
        If x in S: L \cap S = {x}.
        If x in N: L \cap S = empty.
        """
        # For speed, we might want to precompute something, but S changes.
        # We can iterate through all points and check lines.
        
        # To make it faster, we can check the "hardest" points first?
        # Or just check all.
        # 2401 points * 50 lines * 49 points/line = ~6M ops.
        # Python might take ~5-10s.
        # If we do this for many candidates, it might be slow.
        
        # Optimization: 
        # If x in S, we need a line L such that no other point of S is on L.
        # If x in N, we need a line L such that no point of S is on L.
        
        # Let's just run the full check.
        
        for x in all_points:
            xa, xb, ya, yb = x
            x_in_S = x in S_set
            
            # Check all 50 lines through x
            found_valid_line = False
            
            # Directions in F_49
            # 1. Vertical: v = (0, 1) -> (0, 0, 0, 1) in (xa, xb, ya, yb) coords?
            # Wait, F_49 is represented by (a, b).
            # v = (v_x, v_y). v_x = (0, 0) is zero vector, not allowed.
            # Vertical line: x = constant.
            # Points: (xa, xb, ya, yb) where xa, xb fixed? No.
            # x coordinate is (xa, xb).
            # Vertical line: x = x0. So xa=xa0, xb=xb0.
            # y varies.
            
            # Line 1: x = x0 (Vertical)
            # Points on line: (xa, xb, ya, yb) for all ya, yb.
            # Check intersection with S.
            if x_in_S:
                # Need L \cap S = {x}
                # Check if any other point in S is on this line.
                # Other points have same xa, xb but different (ya, yb).
                # We can check if there is any point in S with same xa, xb and (ya, yb) != (ya0, yb0).
                # This is a set lookup.
                # To do this efficiently, we should index S by (xa, xb).
                pass
            else:
                # Need L \cap S = empty.
                # Check if any point in S has same xa, xb.
                pass
            
            # This approach requires precomputed indices of S.
            # Since S is fixed during the check, we can build indices.
            
            # Let's rebuild indices for S_set inside the function?
            # That's expensive if called many times.
            # But we only call it a few times.
            
            # Let's implement a helper that builds indices for S.
            
            # Actually, let's just use the set of points in S and iterate?
            # 49 points per line.
            # 50 lines.
            # 2401 points.
            # 2401 * 50 * 49 = 5.8M iterations.
            # Each iteration: check if point is in S_set.
            # Set lookup is O(1).
            # 5.8M lookups is fast in Python (<1s).
            
            # So we can just iterate.
            
            # Directions:
            # v_x, v_y in F_49, not both 0.
            # We can iterate m in F_49 for slope m = v_y / v_x?
            # If v_x = 0, it's vertical.
            # If v_x != 0, we can normalize v_x = 1.
            # So slopes m in F_49 (49 values) + Vertical (1 value).
            
            # F_49 elements: (a, b)
            
            # 1. Vertical Line
            # Points: (xa, xb, ya, yb)
            # Check intersection with S_set.
            # We need to know if (xa, xb, ya, yb) is in S_set.
            
            # Let's just loop t in 0..48
            # Point on line: x + t*v.
            # v = (0, 0, 0, 1) ? No.
            # v_x = (0,0) is not a valid direction for vertical?
            # Vertical line means x is constant.
            # So x(t) = x0. y(t) = y0 + t*(1,0)?
            # In F_49, y = ya + yb*alpha.
            # Vertical line: x = x0.
            # Points: (x0, y) for all y in F_49.
            
            # Let's generate the 50 lines explicitly for a given x.
            
            # Line 0: Vertical
            # v_x = (0,0) is degenerate.
            # We define vertical line as x = x0.
            # Points: (xa, xb, ya, yb)
            
            # Lines 1-49: Slope m = (ma, mb)
            # v_x = (1, 0) ? No, v_x must be non-zero in F_49.
            # Let v_x = (1, 0) which is 1 in F_49.
            # v_y = m.
            # Point(t) = x0 + t * (1, m)
            # x(t) = x0 + t
            # y(t) = y0 + t*m
            
            # We need to check if there exists t such that all points except x0 are in N.
            # i.e., (Point(t) \ {x0}) subset N.
            # i.e., Point(t) not in S for all t != 0.
            
            # So for a given line, we check all 48 other points.
            # If any is in S, the line is invalid for x (if x in S, we allow x in S, but no others).
            # If x in S, we check if any OTHER point is in S.
            # If x in N, we check if ANY point is in S.
            
            # Implementation:
            
            # Precompute list of all 48 other points for each of the 50 lines?
            # No, generate on the fly.
            
            # F_49 addition and multiplication.
            # (a1, b1) + (a2, b2) = (a1+a2, b1+b2) mod 7
            # (a1, b1) * (a2, b2) = (a1*a2 + 3*b1*b2, a1*b2 + a2*b1) mod 7
            
            def add_f49(p1, p2):
                return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
            
            def mul_f49(p1, p2):
                a1, b1 = p1
                a2, b2 = p2
                return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
            
            def add_scalar_f49(p, s):
                # s is int 0..6
                return ((p[0] + s) % p, p[1]) # No, s is in F_49?
                # t is in F_49.
                # t = (ta, tb)
                # t * v_x.
                pass
            
            # This is getting complex.
            # Let's use numpy? No, standard python is fine for 6M ops.
            
            # Let's optimize by checking only a few lines?
            # No, we must check if ANY line works.
            
            # Let's write the check carefully.
            
            # We can precompute the 50 lines for each point?
            # 2401 * 50 * 49 points = 5.8M points.
            # Storing them is ~50MB.
            # Maybe just generate.
            
            # Let's try to add points to S greedily and check validity.
            # Start with S = {p in U | ya in {3,5,6}}
            # Try adding p in U \ S.
            # Check validity.
            # If valid, keep.
            
            # This might take 196 * 1s = 196s.
            # Within 600s limit.
            
            pass

    # Let's implement the greedy addition.
    
    # Initial S
    S = set()
    for pt in points_in_U:
        xa, xb, ya, yb = pt
        if ya in {3, 5, 6}:
            S.add(pt)
            
    # Candidates to add
    candidates = [pt for pt in points_in_U if pt not in S]
    
    # Shuffle candidates? Or sort by ya?
    # Let's sort by ya to be deterministic.
    candidates.sort(key=lambda x: (x[2], x[3], x[0], x[1]))
    
    # Validity Check Function
    def check_validity(S_set):
        # S_set is a set of tuples
        # For each x in all_points:
        #   if x in S_set:
        #     need line L s.t. L \ {x} disjoint from S_set
        #   else:
        #     need line L s.t. L disjoint from S_set
        
        # To speed up, we can break early if we find a point with no valid line.
        
        # Precompute S_set as a set for O(1) lookup.
        
        for x in all_points:
            xa, xb, ya, yb = x
            x_in_S = x in S_set
            
            valid_line_found = False
            
            # Check 50 lines
            # 1. Vertical
            # Points: (xa, xb, ya, yb)
            # We need to check if there is any point in S_set on this line (excluding x if x in S).
            # Line points: (xa, xb, ya', yb') for all ya', yb'.
            # If x in S:
            #   Check if exists (xa, xb, ya', yb') in S_set such that (ya', yb') != (ya, yb).
            # If x in N:
            #   Check if exists (xa, xb, ya', yb') in S_set.
            
            # This requires iterating over 49 points.
            # Or we can index S_set by (xa, xb).
            
            # Let's build an index for S_set.
            # index[(xa, xb)] = set of (ya, yb) in S.
            
            # Building index takes O(|S|).
            # 2401 * O(1) + 50 * O(1) per point?
            # If we build index once per check_validity call:
            # O(|S|) + 2401 * 50 * O(1).
            
            # Let's do that.
            
            index = {}
            for s_pt in S_set:
                s_xa, s_xb, s_ya, s_yb = s_pt
                key = (s_xa, s_xb)
                if key not in index:
                    index[key] = set()
                index[key].add((s_ya, s_yb))
            
            # Now check each point x
            for x in all_points:
                xa, xb, ya, yb = x
                x_in_S = x in S_set
                y_tuple = (ya, yb)
                
                valid_line_found = False
                
                # 1. Vertical Line
                # Key (xa, xb)
                key = (xa, xb)
                s_points_on_v_line = index.get(key, set())
                
                if x_in_S:
                    # Need s_points_on_v_line == {y_tuple}
                    if s_points_on_v_line == {y_tuple}:
                        valid_line_found = True
                else:
                    # Need s_points_on_v_line == empty
                    if not s_points_on_v_line:
                        valid_line_found = True
                
                if valid_line_found:
                    continue # Next x
                
                # 2. Slanted Lines
                # m in F_49
                # v_x = (1, 0)
                # v_y = m = (ma, mb)
                # Points: x + t*v
                # x(t) = x + t
                # y(t) = y + t*m
                
                # We need to check if there is any t != 0 such that x(t) in S.
                # If x in S, we allow t=0 (x itself), but no other t.
                # If x in N, we allow no t.
                
                # We can iterate t in 1..48.
                # For each t, calculate x(t), y(t).
                # Check if (x(t), y(t)) in S.
                
                # To make this faster, we can check if the line intersects S.
                # Instead of iterating t, we can iterate over points in S and see if they lie on the line?
                # No, |S| is large.
                
                # Iterating t is 48 steps.
                # 50 lines * 48 steps = 2400 steps per x.
                # 2401 x * 2400 = 5.7M steps.
                # Each step involves F_49 addition/multiplication and set lookup.
                # This is acceptable.
                
                for ma in range(p):
                    for mb in range(p):
                        if ma == 0 and mb == 0:
                            continue # v_x = (1,0) is fixed, v_y=(0,0) is horizontal?
                            # Wait, if v_y = (0,0), line is y = constant.
                            # This is covered by m=0.
                        
                        # m = (ma, mb)
                        # v_x = (1, 0)
                        # v_y = (ma, mb)
                        
                        # Check if line intersects S (excluding x if x in S)
                        
                        line_has_other_S = False
                        if not x_in_S:
                            # Check if ANY point on line is in S
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # t = (ta, tb)
                                    # x_t = x + t*v_x = x + t*(1,0)
                                    # y_t = y + t*v_y = y + t*(ma, mb)
                                    
                                    # t * (1,0) = (ta, 0)
                                    # x_t = (xa + ta, xb)
                                    
                                    # t * (ma, mb)
                                    # (ta, tb) * (ma, mb)
                                    # = (ta*ma + 3*tb*mb, ta*mb + tb*ma)
                                    
                                    tx = (ta * ma + 3 * tb * mb) % p
                                    ty = (ta * mb + tb * ma) % p
                                    
                                    # y_t = (ya + tx, yb + ty)
                                    yta = (ya + tx) % p
                                    ytb = (yb + ty) % p
                                    
                                    # Point: (xa+ta, xb, yta, ytb)
                                    pt_t = ((xa + ta) % p, xb, yta, ytb)
                                    
                                    if pt_t in S_set:
                                        line_has_other_S = True
                                        break
                                if line_has_other_S:
                                    break
                            
                            if not line_has_other_S:
                                valid_line_found = True
                                break
                        else:
                            # x in S
                            # Check if ANY OTHER point on line is in S
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    
                                    tx = (ta * ma + 3 * tb * mb) % p
                                    ty = (ta * mb + tb * ma) % p
                                    
                                    yta = (ya + tx) % p
                                    ytb = (yb + ty) % p
                                    
                                    pt_t = ((xa + ta) % p, xb, yta, ytb)
                                    
                                    if pt_t in S_set:
                                        line_has_other_S = True
                                        break
                                if line_has_other_S:
                                    break
                            
                            if not line_has_other_S:
                                valid_line_found = True
                                break
                    
                    if valid_line_found:
                        break
                
                if not valid_line_found:
                    return False
                    
            return True

    # Greedy Addition
    # We have 196 candidates.
    # Each check takes ~1-2s.
    # Total ~200-400s.
    # Safe.
    
    for pt in candidates:
        S.add(pt)
        if check_validity(S):
            # Keep
            pass
        else:
            S.remove(pt)
            
    # Convert S to N
    N = []
    for pt in all_points:
        if pt not in S:
            N.append([[pt[0], pt[1]], [pt[2], pt[3]]])
            
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
