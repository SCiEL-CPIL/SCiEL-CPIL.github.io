import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Separable Multiplicative Polynomial Basis.
    
    Model: f(x, y, z) = P_x(x) * P_y(y) * P_z(z)
    
    Where:
    P_x(x) = a[0] + a[1]*x + a[2]*x^2
    P_y(y) = a[3] + a[4]*y + a[5]*y^2
    P_z(z) = a[6] + a[7]*z
    
    This form allows for rich multiplicative interactions between the dimensions.
    For example, if the target function is exp(x+y+z), the quadratic terms can approximate 
    the exponential growth locally, and the multiplicative structure captures the 
    separability. Even if the function isn't perfectly separable, this basis provides 
    a different hypothesis space than additive polynomials or RBFs, potentially 
    escaping local minima or fitting specific structures better.
    
    Parameters (10 total):
    p[0:3] -> Coefficients for x polynomial (constant, linear, quadratic)
    p[3:6] -> Coefficients for y polynomial (constant, linear, quadratic)
    p[6:8] -> Coefficients for z polynomial (constant, linear)
    p[8:10]-> Unused in this specific parametrization, but kept for interface compliance.
              Alternatively, we could use p[8] and p[9] to extend z to quadratic or 
              add a global scale/offset. Let's extend z to quadratic to use all params 
              symmetrically? 
              Actually, let's stick to the prompt's suggestion: (a0+a1x+a2x^2)(b0+b1y)(c0+c1z).
              That uses 2+1+1 = 4 params? No, the prompt says:
              (a0 + a1 x + a2 x^2)(b0 + b1 y)(c0 + c1 z)
              This uses:
              x: 3 params
              y: 2 params
              z: 2 params
              Total: 7 params.
              We have 10 params. 
              
              Let's refine the basis to use all 10 parameters meaningfully.
              Option A: (a0 + a1 x + a2 x^2)(b0 + b1 y + b2 y^2)(c0 + c1 z + c2 z^2)
              This uses 3+3+3 = 9 params. One left over.
              Option B: Add a global offset? No, must be closed form with params.
              Option C: Use the 10th param as a scale factor?
              
              Let's go with Option A plus a scale factor or just use 9 and ignore the last?
              Better: Let's use the 10th param as a multiplier for the whole product?
              Or, let's make z quadratic too, and use the 10th param as a constant offset added to the product?
              
              Let's try: 
              f(x,y,z) = S * (Ax0 + Ax1*x + Ax2*x^2) * (Ay0 + Ay1*y + Ay2*y^2) * (Az0 + Az1*z + Az2*z^2)
              This uses 1 + 3 + 3 + 3 = 10 params.
              
              Params:
              p[0]: Scale S
              p[1:4]: x coeffs
              p[4:7]: y coeffs
              p[7:10]: z coeffs
    """
    start_time = time.time()
    
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Ensure p has at least 10 elements, pad with zeros if needed
        if len(p) < 10:
            p = np.pad(p, (0, 10 - len(p)))
        else:
            p = p[:10]
            
        S = p[0]
        
        # x polynomial: Ax0 + Ax1*x + Ax2*x^2
        ax0, ax1, ax2 = p[1], p[2], p[3]
        px = ax0 + ax1 * x + ax2 * x * x
        
        # y polynomial: Ay0 + Ay1*y + Ay2*y^2
        ay0, ay1, ay2 = p[4], p[5], p[6]
        py = ay0 + ay1 * y + ay2 * y * y
        
        # z polynomial: Az0 + Az1*z + Az2*z^2
        az0, az1, az2 = p[7], p[8], p[9]
        pz = az0 + az1 * z + az2 * z * z
        
        output = S * px * py * pz
        
        # Safety check for non-finite values
        if not np.all(np.isfinite(output)):
             # Return zeros if any non-finite values are produced
             return np.zeros_like(x)
             
        return output
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
