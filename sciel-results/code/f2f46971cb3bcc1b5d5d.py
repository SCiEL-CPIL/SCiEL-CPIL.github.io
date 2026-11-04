"""Physics-based hexagon packing: rigid-body dynamics with shrinking container."""
import numpy as np
import time
from math import cos, sin, pi, sqrt, atan2, floor

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon (side s) centered at (cx,cy) rotated ang_deg."""
    a = np.radians(ang_deg)
    ca, sa = cos(a), sin(a)
    base = np.array([[cos(2*pi*k/6), sin(2*pi*k/6)] for k in range(6)]) * s
    R = np.array([[ca, -sa], [sa, ca]])
    return base @ R.T + np.array([cx, cy])

def _poly_overlap_area(v1, v2):
    """Approximate overlap area via SAT-like separation; return 0 if separated, else positive."""
    # Use a simple check: if any separating axis exists, no overlap.
    # For convex polygons, check all edge normals.
    def axes(v):
        n = len(v)
        ax = []
        for i in range(n):
            e = v[(i+1) % n] - v[i]
            ax.append(np.array([-e[1], e[0]]))
        return ax
    ax1 = axes(v1)
    ax2 = axes(v2)
    for ax in ax1 + ax2:
        ax_norm = ax / (np.linalg.norm(ax) + 1e-15)
        p1 = v1 @ ax_norm
        p2 = v2 @ ax_norm
        if p1.max() < p2.min() or p2.max() < p1.min():
            return 0.0, 0.0, 0.0  # separated
    # Estimate overlap depth as min penetration along axes
    min_pen = float('inf')
    best_axis = None
    for ax in ax1 + ax2:
        ax_norm = ax / (np.linalg.norm(ax) + 1e-15)
        p1 = v1 @ ax_norm
        p2 = v2 @ ax_norm
        pen = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if pen < min_pen:
            min_pen = pen
            best_axis = ax_norm
    return min_pen, best_axis[0], best_axis[1]

def _hexagon_in_container(v, cx, cy, S, ang_deg):
    """Check if hexagon vertices v are all inside container hexagon. Return True if inside."""
    a = np.radians(ang_deg)
    ca, sa = cos(a), sin(a)
    # Container vertices
    cont = np.array([[cos(2*pi*k/6), sin(2*pi*k/6)] for k in range(6)]) * S
    R = np.array([[ca, -sa], [sa, ca]])
    cont = cont @ R.T + np.array([cx, cy])
    # For each container edge, check all vertices are on the correct side
    for i in range(6):
        e1 = cont[i]
        e2 = cont[(i+1) % 6]
        edge = e2 - e1
        normal = np.array([-edge[1], edge[0]])
        # Ensure normal points inward
        center_proj = (np.array([cx, cy]) - e1) @ normal
        if center_proj < 0:
            normal = -normal
        for pt in v:
            if (pt - e1) @ normal < -1e-9:
                return False
    return True

def _max_dist_to_container(v, cx, cy, S, ang_deg):
    """Find how much v needs to shrink or move to be inside container. Return max violation."""
    a = np.radians(ang_deg)
    ca, sa = cos(a), sin(a)
    cont = np.array([[cos(2*pi*k/6), sin(2*pi*k/6)] for k in range(6)]) * S
    R = np.array([[ca, -sa], [sa, ca]])
    cont = cont @ R.T + np.array([cx, cy])
    max_viol = 0.0
    for i in range(6):
        e1 = cont[i]
        e2 = cont[(i+1) % 6]
        edge = e2 - e1
        normal = np.array([-edge[1], edge[0]])
        center_proj = (np.array([cx, cy]) - e1) @ normal
        if center_proj < 0:
            normal = -normal
        for pt in v:
            d = (pt - e1) @ normal
            if d < -1e-9:
                max_viol = max(max_viol, -d)
    return max_viol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial configuration: place hexagons in a rough grid
    # Side length 1, so diameter ~2, need spacing >= ~2
    # Start with a 4x3 grid
    cols, rows = 4, 3
    spacing = 2.1
    positions = []
    angles = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols-1)/2.0) * spacing
            y = (r - (rows-1)/2.0) * spacing * 0.866  # hexagonal packing offset
            positions.append([x, y])
            angles.append(0.0)
    
    positions = np.array(positions, dtype=float)
    angles = np.array(angles, dtype=float)
    
    # Physics simulation with shrinking container
    dt = 0.005
    friction = 0.95
    k_repel = 50.0
    k_cont = 200.0
    damping = 0.92
    
    # Start with a large container
    S_current = 8.0
    S_min = 2.5  # theoretical lower bound
    
    velocities = np.zeros_like(positions)
    
    # Track best solution found
    best_score = float('inf')
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_S = S_current
    best_center = np.array([0.0, 0.0])
    
    # Simulation loop
    step = 0
    while time.time() < deadline - 20.0:  # leave 20s for polish
        step += 1
        
        # Slowly shrink the container
        if S_current > S_min + 0.01:
            S_current = max(S_min, S_current * 0.9995)
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion between hexagons
        for i in range(N):
            vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i+1, N):
                vj = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                # Quick distance check
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist > 4.0:
                    continue
                pen, axx, axy = _poly_overlap_area(vi, vj)
                if pen > 0:
                    # Repulsive force along axis
                    f = k_repel * pen
                    forces[i] += f * np.array([axx, axy])
                    forces[j] -= f * np.array([axx, axy])
        
        # Container forces: push hexagons inward if they violate
        for i in range(N):
            vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            viol = _max_dist_to_container(vi, 0.0, 0.0, S_current, 0.0)
            if viol > 0:
                # Push toward center
                r = np.linalg.norm(positions[i])
                if r > 1e-6:
                    dir_to_center = -positions[i] / r
                else:
                    dir_to_center = np.array([1.0, 0.0])
                forces[i] += k_cont * viol * dir_to_center
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        velocities *= friction
        positions += velocities * dt
        
        # Keep positions bounded
        positions[:, 0] = np.clip(positions[:, 0], -S_current, S_current)
        positions[:, 1] = np.clip(positions[:, 1], -S_current, S_current)
        
        # Every 500 steps, try to find the minimal container that fits
        if step % 500 == 0:
            # Binary search for minimal S
            lo, hi = 2.0, S_current + 1.0
            for _ in range(30):
                mid = (lo + hi) / 2.0
                ok = True
                for i in range(N):
                    vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                    if not _hexagon_in_container(vi, 0.0, 0.0, mid, 0.0):
                        ok = False
                        break
                if ok:
                    hi = mid
                else:
                    lo = mid
            if hi < best_score:
                # Check for overlaps
                overlap_free = True
                for i in range(N):
                    vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                    for j in range(i+1, N):
                        vj = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                        pen, _, _ = _poly_overlap_area(vi, vj)
                        if pen > 1e-6:
                            overlap_free = False
                            break
                    if not overlap_free:
                        break
                if overlap_free:
                    best_score = hi
                    best_positions = positions.copy()
                    best_angles = angles.copy()
                    best_S = hi
                    best_center = np.array([0.0, 0.0])
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Final polish: local optimization to reduce container size
    if time.time() < deadline - 5.0:
        # Try small perturbations and local search
        pos = best_positions.copy()
        ang = best_angles.copy()
        S = best_S
        
        for iteration in range(500):
            if time.time() > deadline - 3.0:
                break
            
            # Compute current minimal container
            lo, hi = max(2.0, S - 0.5), S + 0.5
            for _ in range(25):
                mid = (lo + hi) / 2.0
                ok = True
                for i in range(N):
                    vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                    if not _hexagon_in_container(vi, 0.0, 0.0, mid, 0.0):
                        ok = False
                        break
                if ok:
                    hi = mid
                else:
                    lo = mid
            
            # Check overlaps
            overlap_free = True
            for i in range(N):
                vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                for j in range(i+1, N):
                    vj = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, ang[j])
                    pen, _, _ = _poly_overlap_area(vi, vj)
                    if pen > 1e-6:
                        overlap_free = False
                        break
                if not overlap_free:
                    break
            
            if overlap_free and hi < S:
                S = hi
                if S < best_score:
                    best_score = S
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_S = S
                    best_center = np.array([0.0, 0.0])
            
            # Small random perturbation
            if S > 2.5 + 0.01:
                scale = 0.05 * (S - 2.5)
                pos += np.random.randn(N, 2) * scale
                ang += np.random.randn(N) * 5.0 * np.pi / 180.0
            else:
                scale = 0.01
                pos += np.random.randn(N, 2) * scale
    
    # Return the best solution found
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_S), 0.0

# EVOLVE-BLOCK-END
