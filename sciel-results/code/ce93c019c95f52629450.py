import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed-Basis Linear Regression with Orthogonal Legendre Polynomials.
    
    The 10 parameters are linear coefficients for a basis of 10 orthogonal polynomial terms.
    By mapping inputs from [0.5, 2] to [-1, 1], we use Legendre polynomials P_0, P_1, P_2.
    
    Basis functions (orthogonal over [-1,1]^3):
    1. P0(x)P0(y)P0(z)  -> 1
    2. P1(x)P0(y)P0(z)  -> x'
    3. P0(x)P1(y)P0(z)  -> y'
    4. P0(x)P0(y)P1(z)  -> z'
    5. P2(x)P0(y)P0(z)  -> 0.5*(3x'^2 - 1)
    6. P0(x)P2(y)P0(z)  -> 0.5*(3y'^2 - 1)
    7. P0(x)P0(y)P2(z)  -> 0.5*(3z'^2 - 1)
    8. P1(x)P1(y)P0(z)  -> x'y'
    9. P1(x)P0(y)P1(z)  -> x'z'
    10. P0(x)P1(y)P1(z) -> y'z'
    
    This formulation makes the optimization problem strictly convex (linear in params),
    allowing BFGS to find the global minimum in 1-2 steps, avoiding the local minima
    traps of the RBF approach.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Map inputs from [0.5, 2] to [-1, 1]
        # x' = 2 * (x - 1.25) / (2 - 0.5) = (x - 1.25) / 0.75
        # Let's use a simpler mapping: x' = 2*x - 2.5? No, range is 0.5 to 2.
        # Center is 1.25. Half-width is 0.75.
        # x_norm = (x - 1.25) / 0.75
        # If x=0.5, x_norm = -0.75/0.75 = -1
        # If x=2, x_norm = 0.75/0.75 = 1
        
        xn = (x - 1.25) / 0.75
        yn = (y - 1.25) / 0.75
        zn = (z - 1.25) / 0.75
        
        # Legendre Polynomials
        # P0(t) = 1
        # P1(t) = t
        # P2(t) = 0.5 * (3*t^2 - 1)
        
        P0_x = np.ones_like(xn)
        P1_x = xn
        P2_x = 0.5 * (3.0 * xn * xn - 1.0)
        
        P0_y = np.ones_like(yn)
        P1_y = yn
        P2_y = 0.5 * (3.0 * yn * yn - 1.0)
        
        P0_z = np.ones_like(zn)
        P1_z = zn
        P2_z = 0.5 * (3.0 * zn * zn - 1.0)
        
        # Construct the 10 basis functions
        # Note: We don't strictly need orthogonality for correctness, but it helps conditioning.
        # The basis is:
        # 1. 1
        # 2. xn
        # 3. yn
        # 4. zn
        # 5. P2_x
        # 6. P2_y
        # 7. P2_z
        # 8. xn * yn
        # 9. xn * zn
        # 10. yn * zn
        
        b1 = P0_x * P0_y * P0_z
        b2 = P1_x * P0_y * P0_z
        b3 = P0_x * P1_y * P0_z
        b4 = P0_x * P0_y * P1_z
        b5 = P2_x * P0_y * P0_z
        b6 = P0_x * P2_y * P0_z
        b7 = P0_x * P0_y * P2_z
        b8 = P1_x * P1_y * P0_z
        b9 = P1_x * P0_y * P1_z
        b10 = P0_x * P1_y * P1_z
        
        # Linear combination
        # output = p[0]*b1 + p[1]*b2 + ... + p[9]*b10
        
        # To be safe against potential numerical issues, we clip the inputs to a reasonable range
        # though they should be within [-1, 1] by construction.
        # We use np.dot for efficiency if we stacked, but manual sum is fine and transparent.
        
        output = (p[0] * b1 + 
                  p[1] * b2 + 
                  p[2] * b3 + 
                  p[3] * b4 + 
                  p[4] * b5 + 
                  p[5] * b6 + 
                  p[6] * b7 + 
                  p[7] * b8 + 
                  p[8] * b9 + 
                  p[9] * b10)
        
        # Safety check
        if not np.all(np.isfinite(output)):
            return np.zeros_like(x)
            
        return output
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
