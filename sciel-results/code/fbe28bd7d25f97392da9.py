"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def lj_energy_and_grad(x):
    """
    Calculate Lennard-Jones energy and gradient for a flat array of coordinates.
    x: shape (3N,)
    """
    N = len(x) // 3
    X = x.reshape(N, 3)
    
    # Compute pairwise distances
    # diff[i, j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(r2, 1.0)
    
    r = np.sqrt(r2)
    r6 = r**-6
    r12 = r6**2
    
    # Energy: 4 * sum(r^-12 - r^-6)
    # We sum over all pairs i < j. 
    # The matrix r12 and r6 includes both (i,j) and (j,i).
    # Sum over all elements gives 2 * sum_{i<j} (r^-12 - r^-6).
    # So energy = 2 * 4 * (sum(r12) - sum(r6)) - 4*N*(1-1) (diagonal is 1-1=0, but we set r2=1 so r12=1, r6=1, diff=0)
    # Wait, diagonal r2=1 => r=1 => r12=1, r6=1 => term is 0.
    # So sum over all matrix elements is 2 * sum_{i<j} (r^-12 - r^-6).
    # Total Energy = 4 * sum_{i<j} (r^-12 - r^-6) = 2 * sum_{all i,j} (r^-12 - r^-6).
    
    E = 2.0 * np.sum(r12 - r6)
    
    # Gradient
    # dE/dx_i = sum_{j != i} dE/dr_ij * dr_ij/dx_i
    # E_pair = 4(r^-12 - r^-6)
    # dE_pair/dr = 4(-12r^-13 + 6r^-7) = -48r^-13 + 24r^-7
    # dr/dx_i = (x_i - x_j) / r
    # dE/dx_i = sum_j (-48r^-13 + 24r^-7) * (x_i - x_j) / r
    # = sum_j (-48r^-14 + 24r^-8) * (x_i - x_j)
    
    # coeff[j, k] = -48 * r[j,k]^-14 + 24 * r[j,k]^-8
    # Note: diagonal is 0 because (x_i - x_i) = 0.
    r14 = r**-14
    r8 = r**-8
    coeff = -48.0 * r14 + 24.0 * r8
    
    # grad[i] = sum_j coeff[i, j] * (X[i] - X[j])
    # This is a matrix multiplication: coeff @ diff
    grad = coeff @ diff
    
    # Flatten gradient
    grad_flat = grad.flatten()
    
    return E, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Generate FCC lattice points
    # FCC points: (i, j, k) * a/2 where i,j,k have same parity
    # Let's generate a sufficient number of points around the origin
    # We need 38 points. The truncated octahedron is roughly spherical.
    # Let's generate points with coordinates in range [-5, 5]
    
    points = []
    # Parity 0: even, even, even
    for i in range(-10, 11, 2):
        for j in range(-10, 11, 2):
            for k in range(-10, 11, 2):
                points.append((i, j, k))
    # Parity 1: odd, odd, odd
    for i in range(-10, 11, 2):
        for j in range(-10, 11, 2):
            for k in range(-10, 11, 2):
                points.append((i, j, k))
                
    points = np.array(points, dtype=float)
    
    # Scale factor
    # We want nearest neighbor distance ~ 1.1225 (2^(1/6))
    # In this lattice (before scaling), nearest neighbors are distance 1.0 apart?
    # (0,0,0) to (1,1,0) dist is sqrt(2). (0,0,0) to (2,0,0) dist is 2.
    # Wait, (0,0,0) is even. (1,1,0) is odd? No, 1,1,0 is mixed parity.
    # My generation:
    # Even: 0,0,0. Nearest even neighbor? 2,0,0 (dist 2).
    # Odd: 1,1,1. Nearest odd neighbor? 3,1,1 (dist 2).
    # Nearest even to odd? (0,0,0) to (1,1,1) dist sqrt(3).
    # (0,0,0) to (1,1,0)? (1,1,0) is not generated.
    # Let's check parity.
    # i,j,k same parity.
    # (0,0,0) -> (0,0,0)
    # (0,0,2) -> dist 2
    # (0,2,0) -> dist 2
    # (2,0,0) -> dist 2
    # (1,1,1) -> dist sqrt(3) ~ 1.732
    # (1,1,-1) -> dist sqrt(3)
    # (1,-1,1) -> dist sqrt(3)
    # (-1,1,1) -> dist sqrt(3)
    # So nearest neighbors in this set are at distance sqrt(3) for the center atom?
    # No, (0,0,0) to (1,1,1) is sqrt(3).
    # Is there a closer one?
    # (0,0,0) to (1,1,0) - not in set.
    # (0,0,0) to (2,0,0) - dist 2.
    # So the "nearest" in this sparse lattice is sqrt(3).
    # This is not a dense FCC lattice.
    # Standard FCC: (0,0,0), (0,1,1), (1,0,1), (1,1,0) scaled by 1/2.
    # Let's use the explicit FCC basis vectors to be safe.
    
    # FCC basis:
    # a1 = (1, 1, 0) / 2
    # a2 = (1, 0, 1) / 2
    # a3 = (0, 1, 1) / 2
    # Points: n1*a1 + n2*a2 + n3*a3
    # = (n1/2 + n2/2, n1/2 + n3/2, n2/2 + n3/2)
    # Let u=n1, v=n2, w=n3.
    # x = (u+v)/2, y=(u+w)/2, z=(v+w)/2
    # u = x+y-z, v=y+z-x, w=x+z-y
    # For x,y,z to be half-integers (since u,v,w are ints), x+y-z must be int.
    # This implies x,y,z are all integers or all half-integers with .5?
    # If x,y,z are all integers: u,v,w are ints.
    # If x,y,z are all k+0.5: u,v,w are ints.
    
    # So the set of FCC points is:
    # 1. Integers (i, j, k)
    # 2. Half-integers (i+0.5, j+0.5, k+0.5)
    # Wait, let's verify (1,1,0)/2 = (0.5, 0.5, 0).
    # Is (0.5, 0.5, 0) in "all half-integers"? No, z=0 is integer.
    # So my parity logic was slightly off or needs adjustment.
    
    # Correct FCC set:
    # Points (x,y,z) such that x+y, y+z, z+x are integers.
    # This is equivalent to: x,y,z are all integers OR x,y,z are all half-integers (n+0.5).
    # Let's re-verify (0.5, 0.5, 0).
    # x+y = 1 (int). y+z = 0.5 (not int).
    # So (0.5, 0.5, 0) is NOT in the set defined by "all int or all half-int".
    
    # Let's stick to the basis vector generation to be absolutely correct.
    # Generate n1, n2, n3 in range [-10, 10]
    pts = []
    for n1 in range(-10, 11):
        for n2 in range(-10, 11):
            for n3 in range(-10, 11):
                x = (n1 + n2) * 0.5
                y = (n1 + n3) * 0.5
                z = (n2 + n3) * 0.5
                pts.append([x, y, z])
                
    pts = np.array(pts)
    
    # Remove duplicates (if any, though n1,n2,n3 mapping is 1-to-1)
    # Actually, the mapping is bijective.
    
    # Calculate distances from origin
    dists = np.linalg.norm(pts, axis=1)
    
    # Sort by distance
    indices = np.argsort(dists)
    sorted_pts = pts[indices]
    
    # We need 38 atoms.
    # The truncated octahedron shape for FCC usually includes the first 3 shells.
    # Let's take the first 38 closest points.
    initial_points = sorted_pts[:N].copy()
    
    # Scale to optimal distance
    # The nearest neighbor distance in this unit lattice (a=1 for cubic cell) is sqrt(2)/2 = 1/sqrt(2) ~ 0.707?
    # Let's check distance between (0,0,0) and (0.5, 0.5, 0).
    # (0,0,0) is n1=0,n2=0,n3=0.
    # (0.5, 0.5, 0) is n1=1,n2=1,n3=0 -> x=1, y=0.5, z=0.5. No.
    # n1=1, n2=1, n3=0 -> x=(1+1)/2=1, y=(1+0)/2=0.5, z=(1+0)/2=0.5. Point (1, 0.5, 0.5).
    # Distance from (0,0,0) to (1, 0.5, 0.5) is sqrt(1 + 0.25 + 0.25) = sqrt(1.5) ~ 1.22.
    # Is there a closer one?
    # (0.5, 0.5, 0) requires n1=1, n2=1, n3=0? No, that gave (1, 0.5, 0.5).
    # To get (0.5, 0.5, 0): x=0.5, y=0.5, z=0.
    # (n1+n2)/2 = 0.5 => n1+n2=1
    # (n1+n3)/2 = 0.5 => n1+n3=1
    # (n2+n3)/2 = 0 => n2+n3=0
    # n2=-n3. n1-n3=1, n1+n3=1 => 2n1=2 => n1=1, n3=0, n2=0.
    # So n1=1, n2=0, n3=0 -> x=0.5, y=0.5, z=0.
    # Distance from (0,0,0) to (0.5, 0.5, 0) is sqrt(0.25+0.25) = sqrt(0.5) ~ 0.707.
    # So nearest neighbor distance is 1/sqrt(2).
    
    # We want nearest neighbor distance to be r_min = 2^(1/6) ~ 1.1225.
    # Current NN dist = 0.707.
    # Scale factor s = 1.1225 / 0.707 = 1.587.
    # Note: 2^(1/6) / (1/sqrt(2)) = 2^(1/6) * 2^(1/2) = 2^(4/6) = 2^(2/3) ~ 1.587.
    
    scale = 2.0 ** (2.0 / 3.0)
    initial_points *= scale
    
    # Center of mass
    com = np.mean(initial_points, axis=0)
    initial_points -= com
    
    best_energy = np.inf
    best_positions = initial_points
    
    # Relaxation loop
    # We will try to relax this structure.
    # If it fails or gets stuck, we might try a slightly perturbed version.
    # But for N=38 FCC, this should converge to the global min.
    
    try:
        res = minimize(
            lj_energy_and_grad,
            initial_points.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success:
            final_pos = res.x.reshape(N, 3)
            # Check validity: min distance > 0.5
            # LJ potential is repulsive, so it should be fine.
            if np.all(np.linalg.norm(final_pos - final_pos[0], axis=1) > 0.49):
                 # Just a sanity check, actually need pairwise
                 pass
            
            e, _ = lj_energy_and_grad(res.x)
            if e < best_energy:
                best_energy = e
                best_positions = final_pos
                
    except Exception as e:
        pass

    # If the first attempt didn't yield a very low energy (close to -173), 
    # we might try a few random perturbations, but the FCC seed is very good.
    # Given the time limit, one high-quality relaxation is usually sufficient for this specific "hard case" if the seed is right.
    # The known minimum is -173.928.
    
    if best_energy > -170: # If it didn't converge to the deep well
        # Try a few random perturbations of the FCC seed
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline:
                break
            perturbed = initial_points + rng.normal(0, 0.1, size=(N, 3))
            try:
                res = minimize(
                    lj_energy_and_grad,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.success:
                    e, _ = lj_energy_and_grad(res.x)
                    if e < best_energy:
                        best_energy = e
                        best_positions = res.x.reshape(N, 3)
            except:
                pass

    return best_positions
# EVOLVE-BLOCK-END
