import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Quadratic Response Surface with Interactions.
    
    This model is linear in the 10 parameters, ensuring a convex loss landscape
    for the BFGS optimizer. It captures non-linear behavior through quadratic
    terms (for temperature and pH optima) and interaction terms.
    
    Parameters:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    params: 10 parameters (weights for the basis functions)
    """
    # Deadline guard
    start_time = time.time()
    
    # Normalize inputs to [0, 1] range approximately
    # b: [1, 5] -> [0, 1]
    b_n = (b - 1.0) / 4.0
    # s: [1, 5] -> [0, 1] (training range)
    s_n = (s - 1.0) / 4.0
    # temp: [1, 40] -> [0, 1]
    T_n = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    pH_n = (pH - 2.0) / 10.0
    
    # Clip normalized values to prevent extreme extrapolation issues in OOD, 
    # though BFGS fits on in-distribution data mostly.
    # Clipping helps stability if inputs are slightly out of range.
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    T_n = np.clip(T_n, 0.0, 1.0)
    pH_n = np.clip(pH_n, 0.0, 1.0)
    
    # Construct Basis Functions
    # We use 10 features:
    # 1. Bias
    # 2. Linear b
    # 3. Linear s
    # 4. Linear T
    # 5. Linear pH
    # 6. Quadratic T (captures optimum)
    # 7. Quadratic pH (captures optimum)
    # 8. Interaction b*s
    # 9. Interaction T*pH
    # 10. Interaction b*T
    
    phi = np.stack([
        np.ones_like(b_n),
        b_n,
        s_n,
        T_n,
        pH_n,
        T_n * T_n,
        pH_n * pH_n,
        b_n * s_n,
        T_n * pH_n,
        b_n * T_n
    ], axis=-1)
    
    # Linear combination
    # params is shape (10,)
    # phi is shape (N, 10)
    # Result is shape (N,)
    prediction = phi @ params
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        # Return a safe constant if something went wrong
        return np.zeros_like(b)
        
    return prediction
