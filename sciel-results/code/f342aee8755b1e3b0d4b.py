# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a custom adaptive Dormand-Prince 5(4) method.
    Tight tolerances are used to drive the error as low as possible.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Tolerances
    rtol = 1e-14
    atol = 1e-16
    
    # Step size parameters
    h = (t1 - t0) / 100.0
    min_h = 1e-12
    max_h = 0.1
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    # Dormand-Prince 5(4) coefficients
    c = np.array([0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0])
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    ])
    b5 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    b4 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])
    
    t = t0
    call_count = 0
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h
            
        # Pre-allocate k array for efficiency? Or just compute on the fly.
        # DP5 uses 6 function evaluations. The 7th row of A is for the 4th order check?
        # Actually, standard DP5(4) has 6 stages. The coefficients above are for 7 rows?
        # Let's check the standard DP5(4) tableau.
        # It has 6 stages. c = [0, 1/5, 3/10, 4/5, 8/9, 1].
        # The last stage is shared.
        # The b5 vector has 6 non-zero elements? No, b5 usually has 6 elements.
        # The b4 vector has 6 elements.
        # The array A should be 6x6.
        # My A above is 7x7. This is for an embedded pair that might have a 7th stage for error?
        # No, DP5(4) is 6 stages.
        # Let's use the standard 6-stage DP5.
        
        c6 = np.array([0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0])
        A6 = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0],
            [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0],
            [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0],
            [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0]
        ])
        b5_6 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0])
        b4_6 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0])
        
        k = np.zeros((6, n))
        
        for i in range(6):
            yi = y.copy()
            for j in range(i):
                yi += h * A6[i, j] * k[j]
            k[i] = f(t + c6[i] * h, yi)
            call_count += 1
            
        y5 = y + h * np.dot(b5_6, k)
        y4 = y + h * np.dot(b4_6, k)
        
        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        # Use max norm for step size control as it's more conservative?
        # Or RMS? Standard is often max norm or Euclidean.
        # Let's use max norm to be safe.
        err_norm = np.max(err / scale)
        
        if err_norm <= 1.0:
            # Accept step
            y = y5
            t += h
            if err_norm > 1e-15:
                factor = safety * err_norm ** (-0.2)
                h = min(max_h, max(min_h, h * min(max_factor, max(min_factor, factor))))
            else:
                h = min(max_h, h * max_factor)
        else:
            # Reject step
            factor = safety * err_norm ** (-0.25)
            h = max(min_h, h * min(max_factor, max(min_factor, factor)))
            
        if not np.all(np.isfinite(y)):
            break
            
    return y
# EVOLVE-BLOCK-END
