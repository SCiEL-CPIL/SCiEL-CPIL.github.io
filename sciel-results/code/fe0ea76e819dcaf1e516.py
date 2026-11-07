import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Clamped cubic B-spline basis functions
    # 10 basis functions, degree 3
    # Knots: 4 repeated at start, 6 internal, 4 repeated at end
    # Total 14 knots
    
    internal_knots = np.linspace(-3.0, 3.0, 8)[1:-1]
    knots = np.concatenate((
        np.full(4, -3.0), 
        internal_knots, 
        np.full(4, 3.0)
    ))
    
    n_basis = 10
    degree = 3
    n_knots = len(knots)
    n_points = len(x)
    
    # Recursive computation of B-spline basis functions
    # N[i, j] is the value of the i-th basis function of degree j at all x points
    # We only need to store the current degree level
    
    # Level 0: N_{i, 0}
    # N_{i, 0}(x) = 1 if knots[i] <= x < knots[i+1]
    # We need N_{i, 0} for i=0 to n_knots - 2
    # Max i for N_{i, 3} is 9.
    # N_{9, 3} depends on N_{9, 0}, N_{10, 0}, N_{11, 0}, N_{12, 0}.
    # So we need N_{i, 0} for i up to 12.
    # n_knots = 14, so indices 0..13. N_{i, 0} defined for i=0..12.
    
    N_prev = np.zeros((n_knots - 1, n_points))
    for i in range(n_knots - 1):
        # Handle the last interval to include the right endpoint
        if i == n_knots - 2:
            mask = (x >= knots[i]) & (x <= knots[i+1])
        else:
            mask = (x >= knots[i]) & (x < knots[i+1])
        N_prev[i] = mask.astype(float)
        
    # Iterate for degree 1 to 3
    for d in range(1, degree + 1):
        N_curr = np.zeros((n_knots - 1 - d, n_points))
        for i in range(n_knots - 1 - d):
            # Term 1
            denom1 = knots[i + d] - knots[i]
            if denom1 > 0:
                term1 = (x - knots[i]) / denom1 * N_prev[i]
            else:
                term1 = np.zeros(n_points)
            
            # Term 2
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 > 0:
                term2 = (knots[i + d + 1] - x) / denom2 * N_prev[i + 1]
            else:
                term2 = np.zeros(n_points)
                
            N_curr[i] = term1 + term2
        
        N_prev = N_curr
        
    # N_prev now contains N_{i, 3} for i=0 to 9
    basis = N_prev[:n_basis]
    
    return params @ basis
