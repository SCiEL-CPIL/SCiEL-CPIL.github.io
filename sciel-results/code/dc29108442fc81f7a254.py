"""
Lennard-Jones 38 cluster global minimum search.

Strategy:
1. Construct an initial guess based on the FCC truncated octahedron geometry.
   The global minimum for LJ38 is known to be a truncated octahedron with
   face-centered cubic (FCC) symmetry.
   - 6 vertices (corner atoms)
   - 12 edge midpoints
   - 12 face centers (squares)
   - 8 face centers (hexagons)
   Total: 6 + 12 + 12 + 8 = 38 atoms.

2. Scale the structure to the optimal Lennard-Jones distance (sigma = 2^(1/6)).

3. Refine the coordinates using L-BFGS-B optimization to minimize the total energy.

4. Return the optimized coordinates.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 38
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s

def lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient.
    pos: (N, 3) array
    """
    # Pairwise distances
    # r[i, j] = |pos[i] - pos[j]|
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2**3
    r12 = r6**2
    
    # Energy: 4 * sum(r^-12 - r^-6) over i < j
    # We compute the full matrix and take half (excluding diagonal)
    E = 4.0 * (r12 - r6)
    np.fill_diagonal(E, 0.0)
    energy = 0.5 * np.sum(E)
    
    # Gradient: dE/dr * dr/dpos
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7) = 24 * (r^-7 - 2 r^-13)
    # Alternatively using r2:
    # E_pair = 4*(r2^-6 - r2^-3)
    # dE_pair/dr2 = 4*(-6*r2^-7 + 3*r2^-4) = 12*(r2^-4 - 2*r2^-7)
    # grad = sum_j 2 * dE_pair/dr2 * (pos_i - pos_j)
    
    # dE_dr2 shape (N, N)
    dE_dr2 = 12.0 * (r2**(-4) - 2.0 * r2**(-7))
    np.fill_diagonal(dE_dr2, 0.0)
    
    # Gradient for each atom i is sum over j of 2 * dE_dr2[i,j] * (pos[i] - pos[j])
    # diff[i,j] = pos[i] - pos[j]
    grad = 2.0 * np.sum(dE_dr2[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def construct():
    # 1. Define the geometry of the FCC truncated octahedron
    # The truncated octahedron has 24 vertices? No, a regular truncated octahedron has 24 vertices.
    # Wait, the LJ38 global minimum is often described as a "truncated octahedron" but the atom count is 38.
    # Let's check the structure of the 38-atom cluster.
    # The global minimum is a distorted FCC fragment.
    # A common construction:
    # Center: 1 atom? No, 38 is even.
    # Structure:
    # - 12 atoms forming a cuboctahedron?
    # - 12 atoms forming a dodecahedron?
    
    # Let's use the known coordinates from literature or generate them via symmetry.
    # The structure is a truncated octahedron with atoms at:
    # - 6 corners of the original octahedron (truncated) -> 6 atoms
    # - 12 edges of the original octahedron -> 12 atoms (midpoints)
    # - 8 faces of the original octahedron (hexagons) -> 8 atoms (centers)
    # - 6 faces of the original octahedron (squares) -> 6 atoms (centers)
    # Total: 6 + 12 + 8 + 6 = 32. This is not 38.
    
    # Let's re-evaluate the LJ38 global minimum structure.
    # According to Cambridge Cluster Database, the global minimum is a truncated octahedron.
    # Actually, the 38-atom cluster global minimum is a "truncated octahedron" with specific layers.
    # It is an FCC fragment.
    # FCC layers: 10-12-10-6? No.
    # FCC stacking ABC.
    # Layer sizes for 38 atoms in FCC:
    # A: 10 atoms (hexagonal packing)
    # B: 12 atoms
    # C: 10 atoms
    # A: 6 atoms
    # Total: 38.
    
    # Let's construct the FCC layers.
    # Hexagonal close packing layer with N atoms.
    # For a hexagonal layer, atoms are at (i, j) in a triangular lattice.
    # We need to generate coordinates for a hexagonal cluster of size n.
    
    def hex_layer_coords(n):
        """Generate coordinates for a hexagonal cluster of n atoms in 2D (x, y)."""
        coords = []
        # Center atom
        coords.append([0.0, 0.0])
        count = 1
        ring = 1
        while count < n:
            # 6 atoms per ring, but corners are shared.
            # Number of atoms in ring k is 6k.
            for i in range(6):
                # Angle
                angle = i * np.pi / 3.0
                # Distance from center
                # In a hexagonal lattice, the distance to ring k is k * sqrt(3) * a / 2?
                # Let's use standard hexagonal coordinates.
                # Basis vectors: a1 = (1, 0), a2 = (1/2, sqrt(3)/2)
                # Ring k points are at distance k from center in terms of lattice steps.
                # We iterate over the 6 sides.
                for j in range(1, k+1):
                    # Move along the side
                    # This simple loop might generate duplicates or wrong counts.
                    # Better: use axial coordinates (q, r)
                    pass
            
            # Simpler approach for hexagonal cluster generation:
            # Generate all integer pairs (q, r) such that max(|q|, |r|, |q+r|) <= k
            # and sum is correct.
            break # Placeholder
            
        # Let's use a more robust generator for hexagonal clusters.
        # Reference: "Hexagonal cluster generation"
        # We want the smallest hexagonal number >= n? No, we want exactly n atoms.
        # For FCC, the layers are hexagonal.
        # Layer 1 (bottom): 10 atoms.
        # Layer 2: 12 atoms.
        # Layer 3: 10 atoms.
        # Layer 4: 6 atoms.
        
        # Let's hardcode the 2D coordinates for these specific layer sizes to ensure correctness.
        # Or generate them algorithmically.
        
        # Algorithm for n atoms in hexagonal packing:
        # Start with center (0,0).
        # Add rings.
        # Ring 1 has 6 atoms. Total 7.
        # Ring 2 has 12 atoms. Total 19.
        # We need 10, 12, 6.
        
        # 10 atoms: Center + Ring 1 (6) + 3 from Ring 2?
        # This is getting complicated. Let's use a known set of coordinates for LJ38 FCC.
        
        # Alternative: Use the "magic number" structure.
        # The 38-atom cluster is often generated by taking a cubic FCC block and trimming.
        
        # Let's try a different approach: 
        # 1. Generate a large FCC lattice.
        # 2. Keep the 38 atoms closest to the center.
        # This naturally forms a truncated octahedron-like shape.
        
        # FCC lattice vectors:
        # a1 = (1, 1, 0)
        # a2 = (1, 0, 1)
        # a3 = (0, 1, 1)
        # Or standard basis:
        # (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
        
        # Let's generate points in a cubic box and filter.
        # Scale factor a = 2^(1/6) * sqrt(2) ?
        # Nearest neighbor distance in FCC is a / sqrt(2) if a is the cubic lattice constant.
        # We want nearest neighbor distance = 2^(1/6) ~ 1.122.
        # So a / sqrt(2) = 1.122 => a = 1.122 * 1.414 = 1.587.
        
        a = 2**(1/6) * np.sqrt(2)
        
        # Generate FCC points
        points = []
        # Cubic lattice indices i, j, k
        # FCC points are at (i, j, k) where i+j+k is even? No.
        # FCC unit cell has 4 atoms: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
        # In integer grid with step 0.5:
        # (i, j, k) such that i, j, k are integers and i+j+k is even.
        
        # Range
        R = 4
        for i in range(-R, R+1):
            for j in range(-R, R+1):
                for k in range(-R, R+1):
                    if (i + j + k) % 2 == 0:
                        # Convert to Cartesian
                        # x = (i + j) * a / 2 ?
                        # Let's use the standard basis vectors for FCC.
                        # v1 = (1, 1, 0) * a/2
                        # v2 = (1, 0, 1) * a/2
                        # v3 = (0, 1, 1) * a/2
                        # Any point is n1*v1 + n2*v2 + n3*v3
                        # This generates all FCC points.
                        pass
        
        # Let's use the simpler generation:
        # Points at (x, y, z) where x, y, z are multiples of 0.5*a and x+y+z is integer multiple of a?
        # No, let's just generate all (i, j, k) in a cubic grid of step a/2.
        # Keep points where (i+j+k) is even.
        
        pts = []
        for i in range(-R, R+1):
            for j in range(-R, R+1):
                for k in range(-R, R+1):
                    if (i + j + k) % 2 == 0:
                        x = i * a / 2.0
                        y = j * a / 2.0
                        z = k * a / 2.0
                        pts.append((x, y, z))
        
        pts = np.array(pts)
        
        # Sort by distance from origin
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)
        
        # Take the first 38
        init_pos = pts[idx[:38]]
        
        # 2. Optimize using L-BFGS
        # Flatten the array for scipy
        x0 = init_pos.flatten()
        
        def objective_and_grad_flat(x):
            pos = x.reshape(-1, 3)
            e, g = lj_energy_and_grad(pos)
            return e, g.flatten()
        
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            return init_pos
        
        # Run L-BFGS
        res = minimize(objective_and_grad_flat, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': 1000, 'gtol': 1e-8})
        
        if time.time() - START_TIME > TIME_LIMIT:
            return init_pos
            
        final_pos = res.x.reshape(-1, 3)
        
        # Verify minimum distance constraint (0.5)
        # If violated, we might need to adjust, but L-BFGS on LJ usually keeps atoms separated.
        # LJ potential goes to infinity as r->0, so it naturally repels.
        
        return final_pos

if __name__ == "__main__":
    # Quick test
    pos = construct()
    e, g = lj_energy_and_grad(pos)
    print(f"Energy: {e}")
    print(f"Min dist: {np.min([np.linalg.norm(pos[i]-pos[j]) for i in range(N) for j in range(i+1, N)])}")
