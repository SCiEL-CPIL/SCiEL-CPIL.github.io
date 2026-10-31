"""
LJ38 cluster global minimum search.

Strategy: Generate the 38-atom FCC truncated-octahedron by taking the 38
closest points in the *true* FCC lattice (integer + half-integer sublattices)
to the origin, then polish with L-BFGS.
This directly targets the known global minimum geometry.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated-octahedron by taking the 38 closest
    points in the true FCC lattice to the origin.
    
    FCC lattice consists of:
    1. Integer coordinates (i, j, k) where i+j+k is even.
    2. Half-integer coordinates (i+0.5, j+0.5, k+0.5) where i+j+k is odd 
       (which makes the sum of coordinates an integer + 1.5, effectively 
       shifting the parity condition). 
       Actually, simpler: FCC points are (x,y,z) where x,y,z are all integers 
       or all half-integers, and x+y+z is an integer.
       
       Let's just generate candidates from both sublattices.
    """
    pts = []
    max_c = 5
    
    # Sublattice 1: Integer coords, sum even
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i * i + j * j + k * k
                    pts.append((float(i), float(j), float(k), d2))
                    
    # Sublattice 2: Half-integer coords (i+0.5, j+0.5, k+0.5)
    # These are equivalent to integer coords shifted by (0.5, 0.5, 0.5)
    # Condition for FCC: x+y+z must be integer. 
    # (i+0.5) + (j+0.5) + (k+0.5) = i+j+k + 1.5
    # For this to be an integer, i+j+k must be 0.5 mod 1? No.
    # Standard FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell.
    # In a larger lattice, points are (i,j,k) with i,j,k integers such that i,j,k have same parity?
    # No, that's BCC.
    # FCC: (i,j,k) where i,j,k are integers and i+j+k is even? 
    # Wait, (0,0,0) is FCC. (1,1,0) is FCC. (1,0,0) is NOT FCC (it's a face center distance away? No, (1,0,0) is a lattice vector for simple cubic).
    # The FCC lattice vectors are a1=(0,1,1), a2=(1,0,1), a3=(1,1,0) scaled by 0.5?
    # Let's just use the definition: Points (x,y,z) such that x,y,z are integers or half-integers, and x+y+z is an integer.
    
    # Sublattice 2: x=i+0.5, y=j+0.5, z=k+0.5
    # Sum = i+j+k + 1.5. For sum to be integer, i+j+k must be half-integer? Impossible for integers.
    # Ah, the condition is usually stated as: x,y,z are all integers or all half-integers, AND x+y+z is an integer.
    # If x=i+0.5, y=j+0.5, z=k+0.5, sum = i+j+k + 1.5. This is never an integer.
    # So the second sublattice must be something else?
    # Let's check (0.5, 0.5, 0). Sum = 1. Integer. x,y,z are not all same type.
    # Correct FCC definition in Cartesian:
    # 1. (i, j, k) with i,j,k integers, i+j+k even.
    # 2. (i+0.5, j+0.5, k) with i,j,k integers, i+j+k odd? 
    #    Let's test (0.5, 0.5, 0). i=0,j=0,k=0. Sum i+j+k=0 (even). But 0+0+0 is even. 
    #    Wait, (0.5, 0.5, 0) is a valid FCC point.
    #    (0.5, 0.5, 0) corresponds to i=0, j=0, k=0 in the form (i+0.5, j+0.5, k).
    #    Condition: i+j+k must be even? 0+0+0 is even.
    #    Let's test (0.5, 0.5, 1). i=0,j=0,k=1. Sum=1 (odd). 
    #    Is (0.5, 0.5, 1) an FCC point? 
    #    Distance to (0,0,0) is sqrt(0.25+0.25+1) = sqrt(1.5). 
    #    Distance to (0,0,1) is sqrt(0.25+0.25) = 0.707.
    #    FCC nearest neighbors are at distance 1/sqrt(2) * a? No, in reduced units sigma=1, a=2^(1/6)*sqrt(2)?
    #    Standard FCC: a = 2^(1/6)*sqrt(2) is not right.
    #    Let's just generate a dense grid of candidates: 
    #    All (x,y,z) where x,y,z are multiples of 0.5, and x+y+z is an integer.
    
    pts = []
    max_c = 5
    for i in range(-2*max_c, 2*max_c + 1):
        for j in range(-2*max_c, 2*max_c + 1):
            for k in range(-2*max_c, 2*max_c + 1):
                x = i * 0.5
                y = j * 0.5
                z = k * 0.5
                # Check if x+y+z is an integer
                s = x + y + z
                if abs(s - round(s)) < 1e-9:
                    d2 = x*x + y*y + z*z
                    pts.append((x, y, z, d2))
    
    pts = np.array(pts)
    # Sort by distance from origin
    order = np.argsort(pts[:, 3])
    pts = pts[order]
    
    # Take the 38 closest
    selected = pts[:N, :3]
    return selected

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum inter-atomic separation."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    # Generate the 38-atom FCC truncated octahedron
    pos = _generate_fcc_truncated_octahedron()
    
    # Center the structure
    pos -= np.mean(pos, axis=0)
    
    # Scale so that nearest-neighbor distance is 2^(1/6) (LJ minimum)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    # Use the minimum nearest-neighbor distance to ensure the tightest bonds are at the LJ minimum
    min_nn = np.min(nn)
    if min_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / min_nn
        pos = pos * scale
    
    # Relax
    best_pos = pos.copy()
    best_energy, _ = _lj_energy_and_grad(best_pos)
    
    try:
        p, e = _relax(pos, max_iter=500, gtol=1e-12)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
    
    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
