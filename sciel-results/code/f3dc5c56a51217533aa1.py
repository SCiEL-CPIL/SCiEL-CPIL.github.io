import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic B-spline basis with 10 control points over x in [-3, 3].
    The 10 params are the control-point values.
    Uses clamped knot vector for proper boundary behavior.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([])
    
    # Define clamped cubic B-spline knot vector
    # n_ctrl = 10, degree k = 3
    # Number of knots = n_ctrl + k + 1 = 14
    # Clamped: first k+1=4 knots at -3, last k+1=4 knots at 3
    # Interior knots: 14 - 4 - 4 = 6 knots evenly spaced in (-3, 3)
    
    n_ctrl = 10
    k = 3
    n_knots = n_ctrl + k + 1  # 14
    
    knots = np.zeros(n_knots)
    knots[:k+1] = -3.0  # knots 0-3 = -3
    knots[-(k+1):] = 3.0  # knots 10-13 = 3
    
    # Interior knots: indices 4 through 9 (6 knots)
    # Evenly spaced between -3 and 3, excluding endpoints
    interior_spacing = 6.0 / (n_ctrl - k)  # 6 / 7? No.
    # Actually, for n_ctrl=10, k=3:
    # Interior knots are at indices 4,5,6,7,8,9
    # We want them to span the domain. A common choice is to use
    # linspace(-3, 3, n_ctrl - k + 1)[1:-1] or similar.
    # Let's use n_ctrl - k = 7 intervals? No.
    # Standard uniform interior knots:
    # t_i = -3 + i * (6 / (n_ctrl - k)) for i=1..n_ctrl-k-1?
    # Let's just place 6 interior knots evenly in (-3, 3)
    # But they should align with the control points.
    # Simplest: linspace(-3, 3, n_ctrl - k + 1) gives 7 points.
    # Interior knots are the middle 6? No, 7 points means 6 intervals.
    # Let's use:
    interior_knots = np.linspace(-3.0, 3.0, n_ctrl - k + 1)[1:-1]
    # This gives 5 points? n_ctrl-k+1 = 8. 8 points, [1:-1] gives 6 points.
    # -3, -2.14, -1.28, -0.42, 0.42, 1.28, 2.14, 3
    # [1:-1] -> -2.14, -1.28, -0.42, 0.42, 1.28, 2.14. Correct.
    
    knots[4:10] = interior_knots
    
    # Compute B-spline basis functions using Cox-de Boor recursion
    # B shape: (n, n_ctrl)
    
    # Degree 0
    B = np.zeros((n, n_ctrl))
    for i in range(n_ctrl):
        B[:, i] = np.where((x >= knots[i]) & (x < knots[i+1]), 1.0, 0.0)
    # Handle x == 3.0 case for the last basis function
    B[(x == 3.0), n_ctrl-1] = 1.0
    
    # Degrees 1 to k
    for d in range(1, k + 1):
        new_B = np.zeros((n, n_ctrl))
        for i in range(n_ctrl - d):
            # Term 1: (x - t_i) / (t_{i+d} - t_i) * N_{i, d-1}
            denom1 = knots[i + d] - knots[i]
            if denom1 != 0:
                new_B[:, i] += (x - knots[i]) / denom1 * B[:, i]
            
            # Term 2: (t_{i+d+1} - x) / (t_{i+d+1} - t_{i+1}) * N_{i+1, d-1}
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 != 0:
                new_B[:, i] += (knots[i + d + 1] - x) / denom2 * B[:, i + 1]
        
        B = new_B
    
    # The last basis function N_{n_ctrl-1, k} might be zero at x=3.0 due to strict inequality in lower degrees
    # Let's ensure the partition of unity or at least non-zero at boundaries if needed.
    # Actually, with clamped knots, the basis functions should cover the domain.
    # Let's check if any rows are all zeros (should not happen with clamped knots)
    # and if so, handle it. But BFGS will fail if gradients are zero.
    
    # Compute linear combination
    result = B @ params
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
