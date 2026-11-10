import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is (x, y) where x, y in F_49.
    F_49 element is represented as (a, b) meaning a + b*alpha, with a, b in 0..6.
    
    A Nikodym set N is a set such that for every point x in F_49^2, there exists
    a line L = {x + t*v : t in F_49} with v = (1, m) for some m in F_49, or v = (0, 1),
    such that L \ {x} subset N.
    
    We construct N as the union of lines. If we take a set of lines, then for any point x,
    we need a line through x that is either in our set, or has its other 48 points in N.
    
    Simplest valid construction: Take ALL 49 lines with slope m=0 (horizontal lines y = c for c in F_49).
    Wait, that's not enough. For a point x = (x0, y0), the horizontal line through x is {(x, c) : c in F_49} with x = x0? No.
    A line with direction v = (1, m) through x = (x0, y0) is {(x0 + t, y0 + t*m) : t in F_49}.
    
    Let me reconsider. A valid Nikodym set: for every point p, there's a line through p where all other points are in N.
    
    If N is the entire plane minus a set C (the complement), then for each p in C, we need a line through p where all other points are in N (i.e., not in C). So the line through p can only intersect C at p.
    
    So C must be a set of points such that for each p in C, there exists a line through p that doesn't contain any other point of C.
    
    This means C is a set where no two points are "forced" to be on the same line for all lines through them... wait.
    
    For p in C, we need a line L through p such that L ∩ C = {p}.
    This is equivalent to: for each p in C, there exists a direction v such that the line p + t*v doesn't hit C at any t ≠ 0.
    
    If |C| is small, this is easy. For example, if C is a single point, any line through it works.
    If C has 2 points p1, p2, then for p1, we need a line through p1 not passing through p2. There are 48 such lines. So |C| = 2 works.
    In general, if C is any set where no two points are... wait, for any p in C, the lines through p are 49. The lines through p that also pass through another point q in C are at most |C|-1 (one for each q). So if |C|-1 < 49, i.e., |C| <= 49, then for each p there's a line through p that avoids all other points in C.
    
    So any set C with |C| <= 49 works as the complement! Then N = F_49^2 \ C has size 2401 - |C|.
    
    To maximize the score (complement size), we want |C| as large as possible. So |C| = 49.
    
    Wait, but we need the lines to have direction v = (1, m) or v = (0, 1). These are all 49 possible directions. So yes, for any p, there are 49 lines through p, one for each direction.
    
    So the maximum complement size is 49. Let me verify: if C has 49 points, for each p in C, the number of lines through p that contain another point of C is at most 48 (one for each other point in C, but actually a line is determined by p and q, and different q's might give the same line if they're collinear with p).
    
    Actually, if C is a line of 49 points, then for a point p on this line, ALL 48 other points are on the same line. But there are other lines through p that don't contain any other point of C (since C is just this one line). So for p in C, we can pick a line through p with a different slope. That line will only intersect C at p. So it works!
    
    So C can be any 49 points, as long as for each p in C, there's a line through p not containing any other point of C. This is always true if C is not "too concentrated". In fact, for any set C of size <= 49, for each p in C, at most 48 lines through p contain another point of C (one per other point, but some might share lines). Actually, the number of lines through p that contain at least one other point of C is at most |C|-1 = 48. Since there are 49 lines total, at least 1 line through p contains no other point of C.
    
    So ANY set C of size 49 works! The maximum complement is 49.
    
    Let me pick C to be 49 specific points. For simplicity, let C = {(0, c) : c in F_49}, i.e., the y-axis.
    
    Then N = F_49^2 \ {(0, c) : c in F_49}.
    
    For any point p in N: p is not on the y-axis, so p = (x, y) with x ≠ 0. We need a line through p where all other points are in N. We can pick the vertical line through p? No, vertical line has direction (0,1). The vertical line through p = (x, y) is {(x, t) : t in F_49}. All these points have x ≠ 0, so they're all in N. 
    
    For any point p in C: p = (0, c). We need a line through p where all other points are in N. We can pick a horizontal line through p: {(t, c) : t in F_49}. This line contains p = (0, c) and all other points (t, c) with t ≠ 0, which are in N. 
    
    So N = F_49^2 \ {(0, c) : c in F_49} is a valid Nikodym set with |N| = 2401 - 49 = 2352, and complement size 49.
    
    This is the maximum possible complement size.
    """
    p = 7
    k_const = 3  # alpha^2 = 3
    
    # Complement: the y-axis, i.e., all points (0, y) for y in F_49
    # So N is all points except those with x = (0, 0)
    
    points = []
    for xa in range(p):
        for xb in range(p):
            # If x = 0, skip (this is the y-axis, part of complement)
            if xa == 0 and xb == 0:
                continue
            for ya in range(p):
                for yb in range(p):
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
