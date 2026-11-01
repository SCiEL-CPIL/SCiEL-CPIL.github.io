import numpy as np

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 200 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the common radius.
    Uses a hexagonal grid with optimized row count and spacing.
    """
    best_r = 0
    best_k = 15
    best_d = 0.1
    best_row_lengths = None

    # Try different numbers of rows k
    for k in range(10, 25):
        # Distribute N points into k rows as evenly as possible
        m_min = N // k
        rem = N % k
        # rem rows have m_min + 1 points, k - rem rows have m_min points
        m_max = m_min + 1 if rem > 0 else m_min
        
        # Calculate optimal spacing d
        # Constraints:
        # 1. Width: 2r + (m_max - 1)d <= 1  => r <= 0.5 - 0.5(m_max - 1)d
        # 2. Height: 2r + (k - 1)d * sqrt(3)/2 <= 1 => r <= 0.5 - 0.5(k - 1)sqrt(3)/2 * d
        # 3. Inter-circle: 2r <= d => r <= 0.5d
        
        # We want to maximize r(d) = min(0.5 - A*d, 0.5 - B*d, 0.5*d)
        # where A = 0.5*(m_max - 1), B = 0.5*(k - 1)*sqrt(3)/2
        
        A = 0.5 * (m_max - 1)
        B = 0.5 * (k - 1) * np.sqrt(3) / 2
        
        # Critical points for d:
        # Intersection of 0.5d and 0.5 - A*d: d = 0.5 / (0.5 + A)
        d_width = 0.5 / (0.5 + A) if A > 0 else 1.0
        # Intersection of 0.5d and 0.5 - B*d: d = 0.5 / (0.5 + B)
        d_height = 0.5 / (0.5 + B) if B > 0 else 1.0
        
        # Also consider the boundaries where r becomes 0 (not needed for max, but good for bounds)
        # d_max_width = 1 / (m_max - 1)
        # d_max_height = 2 / ((k - 1) * sqrt(3))
        
        candidates = [d_width, d_height]
        # Add a small epsilon around the intersections to be safe?
        # The maximum of the min function is exactly at one of these intersections
        # provided that at that d, the other constraints are not violated (i.e., the value is <= the others).
        # Actually, since r(d) is concave, the max is at the peak.
        
        best_r_for_k = 0
        best_d_for_k = 0
        
        for d in candidates:
            if d <= 0:
                continue
            r_w = 0.5 - A * d
            r_h = 0.5 - B * d
            r_c = 0.5 * d
            
            r_val = min(r_w, r_h, r_c)
            if r_val > best_r_for_k:
                best_r_for_k = r_val
                best_d_for_k = d
        
        # It is possible that the optimal d is not exactly at the intersection with 0.5d
        # if the two boundary constraints cross each other before crossing 0.5d.
        # Intersection of 0.5 - A*d and 0.5 - B*d is at d=0 if A!=B.
        # So the "kink" is always with the 0.5d line or at d=0.
        # However, if A < B, the width constraint is looser.
        # The function min(0.5 - A*d, 0.5 - B*d, 0.5*d)
        # If A < B, then 0.5 - A*d > 0.5 - B*d.
        # So min is min(0.5 - B*d, 0.5*d).
        # The max is at intersection of 0.5 - B*d and 0.5*d.
        # This is d_height.
        # If B < A, max is at d_width.
        # So checking d_width and d_height is sufficient.
        
        if best_r_for_k > best_r:
            best_r = best_r_for_k
            best_k = k
            best_d = best_d_for_k
            # Store row lengths for reconstruction
            # We need to decide which rows get the extra points.
            # To maintain symmetry, we can put the longer rows in the middle.
            row_lengths = [m_min] * k
            # Distribute 'rem' extra points to the middle rows
            start = (k - rem) // 2
            for i in range(start, start + rem):
                row_lengths[i] = m_min + 1
            best_row_lengths = row_lengths

    # Generate the grid
    d = best_d
    k = best_k
    row_lengths = best_row_lengths
    
    points = []
    for i, m in enumerate(row_lengths):
        # y coordinate for row i
        y = (i - (k - 1) / 2.0) * d * np.sqrt(3) / 2.0
        
        # x coordinates for this row
        # Center the row horizontally
        # Hex shift: odd rows shifted by d/2
        shift = 0.0
        if i % 2 != 0:
            shift = d / 2.0
            
        # Generate x positions
        # We want the row to be centered at 0 (before global shift)
        # x_j = (j - (m-1)/2.0) * d + shift
        xs = (np.arange(m) - (m - 1) / 2.0) * d + shift
        for x in xs:
            points.append([x, y])
            
    points = np.array(points)
    
    # Center the entire grid globally
    # Find min/max x and y
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    points[:, 0] -= center_x
    points[:, 1] -= center_y
    
    return points

# EVOLVE-BLOCK-END
