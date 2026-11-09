import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with normalized inputs and full phase drive.
    
    Model:
    a = p0*X + p1*X^3 + p2*V + p3*V|V| + p4*X^2*V 
      + p5*sin(t)*X + p6*cos(t) + p7*sin(t) + p8*X*V + p9
    
    Where X = x/0.2, V = v/0.3.
    Structural change: Replaced X^2 (quadratic stiffness) with sin(t) (sine forcing)
    to allow arbitrary phase of the periodic drive.
    """
    # Normalization
    X = x / 0.2
    V = v / 0.3
    
    # Powers
    X2 = X * X
    X3 = X2 * X
    
    # Nonlinear damping: V * |V|
    V_abs = np.abs(V)
    V_nonlin = V * V_abs
    
    # Trig
    St = np.sin(t)
    Ct = np.cos(t)
    
    # Extract params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute
    res = (p0 * X + 
           p1 * X3 + 
           p2 * V + 
           p3 * V_nonlin + 
           p4 * X2 * V + 
           p5 * St * X + 
           p6 * Ct + 
           p7 * St + 
           p8 * X * V + 
           p9)
           
    return res
