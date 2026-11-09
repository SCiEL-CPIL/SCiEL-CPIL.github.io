# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by excluding a single line (49 points).
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]] representing (x_a + x_b*alpha, y_a + y_b*alpha).
    
    We exclude all points where y = 0, i.e., y_a = 0 and y_b = 0.
    This gives |S| = 49, so |N| = 2401 - 49 = 2352.
    
    Validity: For any point x:
    - If x is not on the excluded line, any direction works since at most one point 
      on any line through x is in S (and it's not x itself for the parallel direction,
      but other directions are fine).
    - If x is on the excluded line, use a direction not parallel to the line; 
      the line through x in that direction intersects the excluded line only at x,
      so all other points on that line are in N.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Exclude points where y = 0 (ya=0, yb=0)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
