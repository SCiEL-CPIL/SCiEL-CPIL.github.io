"""E. coli growth rate: 4th-order multivariate polynomial in normalized inputs."""
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a 4th-order multivariate polynomial
    in normalized inputs. 10 fitted parameters.
    
    Normalization:
      b_norm = (b - 1) / (4.999 - 1)   -> [0, 1]
      s_norm = (s - 1) / (39.98 - 1)   -> [0, 1]
      t_norm = (temp - 1) / (40 - 1)   -> [0, 1]
      p_norm = (pH - 1) / (14 - 1)     -> [0, 1]
    
    We use a sparse 4th-order polynomial with 10 terms:
      p0 + p1*x0 + p2*x1 + p3*x2 + p4*x3
      + p5*x0*x1 + p6*x0*x2 + p7*x1*x2 + p8*x2*x3 + p9*x0*x1*x2
    where x0=b_norm, x1=s_norm, x2=t_norm, x3=p_norm
    """
    # Normalize inputs to [0, 1] range based on training ranges
    x0 = (b - 1.0) / (4.999 - 1.0)
    x1 = (s - 1.0) / (39.98 - 1.0)
    x2 = (temp - 1.0) / (40.0 - 1.0)
    x3 = (pH - 1.0) / (14.0 - 1.0)
    
    # Clip to avoid extrapolation issues (inputs should be in range)
    x0 = np.clip(x0, 0.0, 1.0)
    x1 = np.clip(x1, 0.0, 1.0)
    x2 = np.clip(x2, 0.0, 1.0)
    x3 = np.clip(x3, 0.0, 1.0)
    
    # Build 10-term polynomial
    # Terms:
    # 0: constant
    # 1: x0
    # 2: x1
    # 3: x2
    # 4: x3
    # 5: x0*x1
    # 6: x0*x2
    # 7: x1*x2
    # 8: x2*x3
    # 9: x0*x1*x2
    result = (
        params[0]
        + params[1] * x0
        + params[2] * x1
        + params[3] * x2
        + params[4] * x3
        + params[5] * x0 * x1
        + params[6] * x0 * x2
        + params[7] * x1 * x2
        + params[8] * x2 * x3
        + params[9] * x0 * x1 * x2
    )
    
    return result
