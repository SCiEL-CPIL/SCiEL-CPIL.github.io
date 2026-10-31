import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    # positions: (N, 3)
    # diff: (N, N, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Sum over all pairs i<j. The matrix sum includes i,j and j,i.
    # Diagonal is 0 in energy terms (4*(1-1)=0), but we filled with 1.0 so 4*(1-1)=0.
    # So sum / 2 is correct.
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron():
    """
    Generate a valid FCC-like cluster of 38 atoms.
    We generate an FCC lattice and pick the 38 closest to the center.
    FCC lattice points can be generated as (i, j, k) where i+j+k is even, scaled.
    Or more simply, use the standard FCC basis in a cubic cell.
    Let's use the integer lattice trick: points (i, j, k) with i+j+k even are BCC in integer coords,
    which corresponds to FCC in physical coords if we scale appropriately?
    Actually, let's just generate a large box of FCC points and cut.
    
    FCC primitive vectors:
    a1 = (0.5, 0.5, 0)
    a2 = (0.5, 0, 0.5)
    a3 = (0, 0.5, 0.5)
    
    Or simpler:
    Lattice points are all (x, y, z) such that x, y, z are integers and x+y+z is even?
    No, that's BCC.
    FCC is:
    (0,0,0) type: (i, j, k) where i,j,k are all even or all odd? No.
    FCC points: (i, j, k) where i, j, k are integers and i+j+k is even?
    Let's check distance between (0,0,0) and (1,1,0). dist^2 = 2.
    (1,1,0) is in the set if 1+1+0=2 (even).
    (1,0,0) is not in the set (1+0+0=1 odd).
    So nearest neighbors are at distance sqrt(2).
    This is a BCC lattice in integer coordinates.
    A BCC lattice with nearest neighbor distance sqrt(2) is equivalent to an FCC lattice with lattice constant a=2?
    Let's verify.
    FCC lattice constant a. Nearest neighbor distance is a/sqrt(2).
    If we set a=2, NN dist is 2/sqrt(2) = sqrt(2).
    So yes, the integer lattice with even sum i+j+k is an FCC lattice with a=2.
    """
    points = []
    # Generate points in a box [-5, 5]
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    points.append((r2, i, j, k))
    
    points.sort(key=lambda x: x[0])
    # Take the first 38
    selected = [p[1:] for p in points[:38]]
    coords = np.array(selected, dtype=float)
    
    # Center
    coords -= np.mean(coords, axis=0)
    
    # Scale to approximate LJ equilibrium.
    # In this lattice, NN dist is sqrt(2).
    # LJ equilibrium r_min = 2^(1/6) * sigma. If sigma=1, r_min = 1.122.
    # Scale factor s = 1.122 / sqrt(2) = 0.792.
    coords *= 0.792
    
    return coords

def make_icosahedron():
    """
    Generate an icosahedral cluster of 38 atoms.
    1 center + 12 first shell + 24 second shell (approx).
    This is a known local minimum trap.
    """
    # Center
    center = np.array([0.0, 0.0, 0.0])
    
    # 12 vertices of an icosahedron.
    # Golden ratio phi
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    # Vertices: (0, +-1, +-phi), (+-1, +-phi, 0), (+-phi, 0, +-1)
    v = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    v = np.array(v)
    # Normalize to radius 1
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    # Scale to LJ radius. Icosahedron NN dist is roughly 1.12?
    # Let's scale so that distance from center to vertex is ~1.12?
    # Actually, for icosahedron, the edge length is the NN distance.
    # Edge length of unit icosahedron (radius 1) is 2*sin(pi/5) ~ 1.175.
    # We want edge length ~ 1.12.
    # Scale factor = 1.12 / 1.175 ~ 0.95.
    shell1 = v * 0.95
    
    # Second shell: 24 atoms.
    # These are typically at the centers of the faces of the icosahedron? No, that's 20.
    # The 2nd shell of the icosahedral cluster (Mackay) has 42 atoms?
    # For N=38, it's 1+12+25? Or 1+12+24?
    # Let's just use a random perturbation of a shell at radius ~2.2 for the remaining 24.
    # Or better, generate random points on a sphere and pick 24.
    rng = np.random.default_rng(42)
    # Generate random points on sphere
    u = rng.random(24)
    v_rand = rng.random(24)
    theta = 2 * math.pi * u
    z = 1 - 2 * v_rand
    r = math.sqrt(1 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shell2 = np.column_stack((x, y, z)) * 2.2
    
    coords = np.vstack([center.reshape(1,3), shell1, shell2])
    return coords

def make_random_cluster():
    """Generate a random cluster."""
    rng = np.random.default_rng(int(time.time()*1000) % 10000)
    coords = rng.uniform(-2, 2, size=(N, 3))
    coords -= np.mean(coords, axis=0)
    return coords

def local_minimize(positions):
    """
    Perform L-BFGS local minimization.
    Returns (positions, energy, success).
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    def grad_func(pos):
        pos = pos.reshape(N, 3)
        # Compute gradient
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, 1.0)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        # Force on i due to j: F_ij = -dU/dr * dr/dx_i
        # U = 4(r^-12 - r^-6)
        # dU/dr = 4(-12 r^-13 + 6 r^-7)
        # Force magnitude F = -dU/dr = 4(12 r^-13 - 6 r^-7)
        # Vector force F_vec = F * (r_i - r_j) / r
        # Gradient of total energy w.r.t r_i is sum of forces? No, gradient is -Force.
        # Wait, minimize energy. Gradient of E w.r.t x_i is sum_j dE/dx_i.
        # dE/dx_i = sum_j dE/dx_ij * dx_ij/dx_i.
        # Let's use numerical gradient or analytic.
        # Analytic:
        # E = sum_{i<j} 4(r_ij^-12 - r_ij^-6)
        # dE/dx_i = sum_{j != i} dE/dx_ij * (-1)  (since x_ij = x_i - x_j)
        # dE/dx_ij = 4(-12 r^-14 * r * e_r + 6 r^-8 * r * e_r) ? No.
        # dE/dr = 4(-12 r^-13 + 6 r^-7)
        # dr/dx_i = (x_i - x_j)/r
        # dE/dx_i = sum_j 4(-12 r^-13 + 6 r^-7) * (x_i - x_j)/r
        #         = sum_j 4(-12 r^-14 + 6 r^-8) * (x_i - x_j)
        
        # Let's compute the term T = 4 * (-12 * r^-14 + 6 * r^-8)
        # T_ij is the coefficient for (x_i - x_j).
        
        inv_r8 = inv_r6 * inv_r2
        inv_r14 = inv_r12 * inv_r2
        coeff = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
        
        # Gradient for atom i is sum_{j != i} coeff_ij * (x_i - x_j)
        # Note: coeff is symmetric? Yes, depends on r.
        # So grad_i = sum_j coeff_ij * (x_i - x_j)
        
        # Vectorized:
        # diff is (N, N, 3) = x_i - x_j
        # coeff is (N, N)
        # grad = sum_j coeff_ij * diff_ij
        # grad_i = np.sum(coeff[:, np.newaxis, :] * diff, axis=1)
        
        grad = np.sum(coeff[:, np.newaxis, :] * diff, axis=1)
        return grad.flatten()
    
    try:
        result = minimize(
            energy_func, 
            positions.flatten(), 
            jac=grad_func,
            method='L-BFGS-B', 
            options={'maxiter': 50, 'gtol': 1e-5, 'ftol': 1e-7}
        )
        final_pos = result.x.reshape(N, 3)
        final_energy = lj_energy_vectorized(final_pos)
        if not check_validity(final_pos):
            return positions, lj_energy_vectorized(positions), False
        return final_pos, final_energy, True
    except Exception:
        return positions, lj_energy_vectorized(positions), False

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_positions = None
    best_energy = float('inf')
    
    # Initial candidates
    candidates = [
        make_fcc_truncated_octahedron(),
        make_icosahedron(),
        make_random_cluster()
    ]
    
    # Pre-relax each candidate to get a good starting point for basin hopping
    for i, cand in enumerate(candidates):
        if time.time() > DEADLINE - 20: break
        pos, energy, _ = local_minimize(cand)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    # Basin Hopping Loop
    # We will perform multiple rounds of kick + relax
    # Kick size: random displacement
    rng = np.random.default_rng(123)
    
    # Estimate time per iteration
    # local_minimize takes ~0.1-0.5s
    # We have ~120s left. Let's allow ~100-200 iterations.
    
    current_pos = best_positions.copy()
    current_energy = best_energy
    
    max_kicks = 150
    kick_size = 0.5
    
    for k in range(max_kicks):
        if time.time() > DEADLINE - 10:
            break
            
        # Kick
        # Randomly perturb positions
        # To avoid breaking the cluster apart, we can perturb a subset or use smaller kicks
        # Standard basin hopping: kick all atoms by random vector
        disp = rng.normal(0, kick_size, size=(N, 3))
        new_pos = current_pos + disp
        
        # Relax
        new_pos, new_energy, success = local_minimize(new_pos)
        
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_pos
        
        # Adaptive kick size?
        # If we haven't improved in a while, increase kick size?
        # Or decrease to refine?
        # Let's keep it simple: constant kick size.
        
    # Final check
    if best_positions is None:
        # Fallback
        best_positions = make_fcc_truncated_octahedron()
        
    # One last relaxation to ensure we are at the local minimum
    final_pos, final_energy, _ = local_minimize(best_positions)
    if final_energy < best_energy:
        best_positions = final_pos
        best_energy = final_energy
        
    # Ensure validity
    if not check_validity(best_positions):
        # If invalid, try to fix by scaling?
        # Or just return the FCC seed
        best_positions = make_fcc_truncated_octahedron()
        # Relax it
        final_pos, final_energy, _ = local_minimize(best_positions)
        if check_validity(final_pos):
             best_positions = final_pos
             
    return best_positions
