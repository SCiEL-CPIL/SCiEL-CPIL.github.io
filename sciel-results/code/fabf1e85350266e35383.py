"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 7 Radial Basis Functions + 3 Polynomial terms.
    Inputs scaled to [-1, 1] for conditioning.
    
    RBF centers (in scaled space [-1,1]^3):
    c0: ( 0.0,  0.0,  0.0)  -> center
    c1: (-0.8, -0.8, -0.8)  -> corner
    c2: ( 0.8,  0.8,  0.8)  -> opposite corner
    c3: (-0.8,  0.8,  0.0)  -> face center
    c4: ( 0.8, -0.8,  0.0)  -> face center
    c5: ( 0.0, -0.8,  0.8)  -> face center
    c6: ( 0.0,  0.8, -0.8)  -> face center
    
    Polynomial terms (in scaled space):
    p7: 1
    p8: xs + ys + zs  (linear trend)
    p9: xs^2 + ys^2 + zs^2  (quadratic trend)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Scale inputs to [-1, 1]: map [0.5, 2] -> [-1, 1]
    # x_scaled = (x - 1.25) / 0.75
    xs = (x - 1.25) / 0.75
    ys = (y - 1.25) / 0.75
    zs = (z - 1.25) / 0.75

    # RBF Centers in scaled coordinates
    cx = np.array([0.0, -0.8, 0.8, -0.8, 0.8, 0.0, 0.0])
    cy = np.array([0.0, -0.8, 0.8, 0.8, -0.8, -0.8, 0.8])
    cz = np.array([0.0, -0.8, 0.8, 0.0, 0.0, 0.8, -0.8])
    
    sigma2 = 0.30  # sigma^2 - slightly tighter for local features

    # Calculate RBF values efficiently using expanded distance formula
    # dist_sq[i,j] = (xs[i]-cx[j])^2 + (ys[i]-cy[j])^2 + (zs[i]-cz[j])^2
    # = (xs^2 + ys^2 + zs^2) - 2*(xs*cx + ys*cy + zs*cz) + (cx^2 + cy^2 + cz^2)
    
    r2 = xs*xs + ys*ys + zs*zs  # (N,)
    
    # Cross term: (N, 7)
    cross = xs[:, np.newaxis] * cx[np.newaxis, :] + \
            ys[:, np.newaxis] * cy[np.newaxis, :] + \
            zs[:, np.newaxis] * cz[np.newaxis, :]
    
    # Center norms: (7,)
    c2 = cx*cx + cy*cy + cz*cz
    
    dist_sq = r2[:, np.newaxis] - 2.0 * cross + c2[np.newaxis, :]
    
    # Clip to avoid negative values due to floating point
    dist_sq = np.maximum(dist_sq, 0.0)
    
    rbf_vals = np.exp(-dist_sq / sigma2)  # (N, 7)
    
    # Polynomial terms (3 terms)
    poly_1 = np.ones_like(x)
    poly_lin = xs + ys + zs
    poly_quad = r2  # xs^2 + ys^2 + zs^2
    
    # Combine: p[0:7] for RBFs, p[7:10] for Polys
    pred = (rbf_vals @ p[0:7] + 
            poly_1 * p[7] + 
            poly_lin * p[8] + 
            poly_quad * p[9])
            
    return pred
# EVOLVE-BLOCK-END
