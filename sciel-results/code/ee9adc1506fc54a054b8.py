import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    npts = x.shape[0]
    
    # 10 cubic B-spline basis functions, clamped on [-3, 3]
    # n=10, k=3, m=14 knots
    # Clamped: t_0=t_1=t_2=t_3=-3, t_10=t_11=t_12=t_13=3
    # Interior knots t_4..t_9: 6 knots evenly spaced in (-3, 3)
    
    interior = np.linspace(-3.0, 3.0, 8)[1:-1]  # 6 values in (-3, 3)
    knots = np.concatenate([np.full(4, -3.0), interior, np.full(4, 3.0)])
    
    # Compute B-spline basis using Cox-de Boor recursion
    # N_i,d for i=0..9, d=0..3
    # At degree d, we have (n + k - d) basis functions? No.
    # At degree 0: basis functions indexed 0..m-2 = 0..12 (13 functions)
    # At degree 1: 0..11 (12 functions)
    # At degree 2: 0..10 (11 functions)
    # At degree 3: 0..9 (10 functions)
    
    # Degree 0
    N = np.zeros((npts, 13))  # 13 intervals for 14 knots
    for i in range(13):
        N[(x >= knots[i]) & (x < knots[i + 1]), i] = 1.0
    # Handle x == knots[-1]
    N[(x == knots[-1]), 12] = 1.0
    
    # Degree 1: 12 basis functions
    N1 = np.zeros((npts, 12))
    for i in range(12):
        d1 = knots[i + 1] - knots[i]
        d2 = knots[i + 2] - knots[i + 1]
        t1 = np.zeros(npts)
        t2 = np.zeros(npts)
        if d1 > 1e-12:
            t1 = (x - knots[i]) / d1 * N[:, i]
        if d2 > 1e-12:
            t2 = (knots[i + 2] - x) / d2 * N[:, i + 1]
        N1[:, i] = t1 + t2
    
    # Degree 2: 11 basis functions
    N2 = np.zeros((npts, 11))
    for i in range(11):
        d1 = knots[i + 2] - knots[i]
        d2 = knots[i + 3] - knots[i + 1]
        t1 = np.zeros(npts)
        t2 = np.zeros(npts)
        if d1 > 1e-12:
            t1 = (x - knots[i]) / d1 * N1[:, i]
        if d2 > 1e-12:
            t2 = (knots[i + 3] - x) / d2 * N1[:, i + 1]
        N2[:, i] = t1 + t2
    
    # Degree 3: 10 basis functions
    N3 = np.zeros((npts, 10))
    for i in range(10):
        d1 = knots[i + 3] - knots[i]
        d2 = knots[i + 4] - knots[i + 1]
        t1 = np.zeros(npts)
        t2 = np.zeros(npts)
        if d1 > 1e-12:
            t1 = (x - knots[i]) / d1 * N2[:, i]
        if d2 > 1e-12:
            t2 = (knots[i + 4] - x) / d2 * N2[:, i + 1]
        N3[:, i] = t1 + t2
    
    return N3 @ params[:10]
