"""Circle packing in unit square n=32 – improved seed via force-directed / gradient ascent."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _initial_hex():
    """Generate a hexagonal-like initial packing of N circles."""
    # Use a hexagonal lattice with spacing s so circles of radius r = s/2 fit
    # We'll place circles in a hex pattern and then take the first N.
    # For n=32, a 6x6 hex grid gives 36 circles; we take 32.
    # Hex lattice: rows offset by half spacing
    # Let's try a grid approach with hexagonal spacing
    # Spacing s, radius r = s/2
    # In a hex lattice, horizontal spacing = s, vertical spacing = s * sqrt(3)/2
    # Let's generate points
    s = 1.0 / 6.0  # 6 columns
    r = s / 2.0
    pts = []
    for row in range(8):
        for col in range(8):
            x = (col + 0.5) * s
            y = (row + 0.5) * s * np.sqrt(3) / 2.0
            if row % 2 == 1:
                x += s / 2.0
            if x < 1.0 and y < 1.0:
                pts.append((x, y))
    # If we don't have enough, add more
    if len(pts) < N:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        s2 = 1.0 / side
        r2 = s2 / 2.0
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append(((i + 0.5) * s2, (j + 0.5) * s2))
    # Take first N
    pts = pts[:N]
    return np.array(pts), r

def _compute_forces(positions, radii, dt=0.001):
    """Compute forces for circle packing: repulsion between overlapping circles,
    attraction to expand radii, and boundary constraints."""
    n = len(positions)
    forces = np.zeros_like(positions)
    d_radii = np.zeros(n)
    
    # Repulsion between circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < 1e-10:
                dx, dy = 1e-5, 1e-5
                dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                # Overlap: push apart
                overlap = min_dist - dist
                fx = overlap * dx / dist
                fy = overlap * dy / dist
                forces[i, 0] -= fx
                forces[i, 1] -= fy
                forces[j, 0] += fx
                forces[j, 1] += fy
                # Also try to reduce radii if overlapping
                d_radii[i] -= 0.5 * overlap
                d_radii[j] -= 0.5 * overlap
    
    # Boundary constraints
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        # Left wall
        if x < r:
            forces[i, 0] += (r - x) * 10
            d_radii[i] -= (r - x) * 5
        # Right wall
        if x > 1 - r:
            forces[i, 0] -= (x - (1 - r)) * 10
            d_radii[i] -= (x - (1 - r)) * 5
        # Bottom wall
        if y < r:
            forces[i, 1] += (r - y) * 10
            d_radii[i] -= (r - y) * 5
        # Top wall
        if y > 1 - r:
            forces[i, 1] -= (y - (1 - r)) * 10
            d_radii[i] -= (y - (1 - r)) * 5
    
    # Attraction to grow radii
    d_radii += 0.1
    
    return forces, d_radii

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Initialize
    positions, base_r = _initial_hex()
    radii = np.full(N, base_r)
    
    best_sum = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    # Simulated annealing / gradient ascent
    lr = 0.01
    temperature = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        forces, d_radii = _compute_forces(positions, radii)
        
        # Update positions
        positions += lr * forces
        # Update radii
        radii += lr * d_radii
        
        # Clamp radii to positive
        radii = np.maximum(radii, 1e-6)
        
        # Clamp positions to stay within bounds (with radius margin)
        for i in range(N):
            radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0],
                          positions[i, 1], 1 - positions[i, 1])
            if radii[i] < 1e-6:
                radii[i] = 1e-6
        
        # Check validity
        valid = True
        for i in range(N):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((positions[i, 0] - positions[j, 0]) ** 2 +
                                  (positions[i, 1] - positions[j, 1]) ** 2)
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # Cool down
        lr *= cooling
        temperature *= cooling
        
        if lr < 1e-8:
            break
    
    # Return best valid result
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_radii
    
    return result
# EVOLVE-BLOCK-END
