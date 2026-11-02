import numpy as np
import time
import math
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x * (sqrt3/2) + y * (1/2)
        L2 = y
        L3 = x * (-sqrt3/2) + y * (1/2)
        L4 = x * (-sqrt3/2) + y * (-1/2)
        L5 = -y
        L6 = x * (sqrt3/2) + y * (-1/2)
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_proj = np.max(all_L)
        return (2.0 / sqrt3) * max_proj

    def generate_lattice_positions(shear_u, shear_v):
        """
        Generate positions for a 3x4 lattice of hexagons with affine shear.
        The lattice basis vectors are:
        e1 = (sqrt(3), 1.5)
        e2 = (0, 3.0)
        Sheared:
        e1' = e1 + shear_u * e2
        e2' = e2 + shear_v * e1
        """
        sqrt3 = math.sqrt(3)
        e1 = np.array([sqrt3, 1.5])
        e2 = np.array([0.0, 3.0])
        
        e1_sheared = e1 + shear_u * e2
        e2_sheared = e2 + shear_v * e1
        
        positions = []
        for i in range(3):
            for j in range(4):
                pos = i * e1_sheared + j * e2_sheared
                positions.append(pos)
        return np.array(positions)

    def objective(shear_params):
        """
        Compute the objective function (outer hexagon side length) for given shear parameters.
        """
        shear_u, shear_v = shear_params
        positions = generate_lattice_positions(shear_u, shear_v)
        return required_outer_side(positions)

    def optimize_lattice(shear_u0, shear_v0, time_budget):
        """
        Optimize the shear parameters using Nelder-Mead or a simple gradient-free method.
        """
        end_time = time.time() + time_budget
        
        # Simple coordinate descent with adaptive step size
        best_u, best_v = shear_u0, shear_v0
        best_side = objective((best_u, best_v))
        
        step_u = 0.1
        step_v = 0.1
        
        while time.time() < end_time:
            improved = False
            
            # Try adjusting shear_u
            for delta in [step_u, -step_u, step_u/2, -step_u/2]:
                new_u = best_u + delta
                new_v = best_v
                side = objective((new_u, new_v))
                if side < best_side:
                    best_u, best_v = new_u, new_v
                    best_side = side
                    improved = True
            
            # Try adjusting shear_v
            for delta in [step_v, -step_v, step_v/2, -step_v/2]:
                new_u = best_u
                new_v = best_v + delta
                side = objective((new_u, new_v))
                if side < best_side:
                    best_u, best_v = new_u, new_v
                    best_side = side
                    improved = True
            
            if not improved:
                step_u *= 0.8
                step_v *= 0.8
                if step_u < 1e-6 and step_v < 1e-6:
                    break
            
            # Occasionally do a larger random jump to escape local minima
            if random.random() < 0.05:
                new_u = best_u + random.uniform(-0.5, 0.5)
                new_v = best_v + random.uniform(-0.5, 0.5)
                side = objective((new_u, new_v))
                if side < best_side:
                    best_u, best_v = new_u, new_v
                    best_side = side
                else:
                    # Accept worse moves with some probability (simulated annealing-like)
                    if random.random() < math.exp(-(side - best_side) / 0.1):
                        best_u, best_v = new_u, new_v
                        best_side = side
        
        return best_u, best_v, best_side

    # Initial guesses for shear parameters
    initial_guesses = [
        (0.0, 0.0),      # No shear
        (0.2, 0.0),      # Positive shear on u
        (-0.2, 0.0),     # Negative shear on u
        (0.0, 0.2),      # Positive shear on v
        (0.0, -0.2),     # Negative shear on v
        (0.1, 0.1),      # Both positive
        (-0.1, -0.1),    # Both negative
        (0.1, -0.1),     # Opposite signs
        (-0.1, 0.1),     # Opposite signs
    ]

    best_positions = None
    best_side = float('inf')
    best_u, best_v = 0.0, 0.0

    # Allocate time to each restart
    num_restarts = len(initial_guesses)
    time_per_restart = (deadline - time.time()) / num_restarts
    if time_per_restart < 0.5:
        time_per_restart = 0.5
        num_restarts = 1

    for i, (u0, v0) in enumerate(initial_guesses):
        if time.time() > deadline:
            break
        
        u_opt, v_opt, side_opt = optimize_lattice(u0, v0, time_per_restart)
        
        if side_opt < best_side:
            best_side = side_opt
            best_u, best_v = u_opt, v_opt
            best_positions = generate_lattice_positions(best_u, best_v)
    
    # Fallback
    if best_positions is None:
        # Use the no-shear lattice
        best_positions = generate_lattice_positions(0.0, 0.0)
        best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
