"""Hexagon packing via simulated annealing with analytic outer-side computation."""
import time
import math
import random
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * math.cos(a + i * math.pi / 3),
                      cy + side * math.sin(a + i * math.pi / 3)] for i in range(6)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    sqrt3 = math.sqrt(3)
    # 6 half-plane constraints for regular hexagon (vertices at angles 0,60,...)
    if lx > s: return False
    if lx < -s: return False
    if abs(sqrt3 * lx + ly) > sqrt3 * s: return False
    if abs(-sqrt3 * lx + ly) > sqrt3 * s: return False
    if abs(-sqrt3 * lx - ly) > sqrt3 * s: return False
    if abs(sqrt3 * lx - ly) > sqrt3 * s: return False
    return True

def _required_side(pos, ang, ocx, ocy, oang):
    """Analytically compute minimum outer side for given inner positions."""
    sqrt3 = math.sqrt(3)
    oa = math.radians(oang)
    ca, sa = math.cos(-oa), math.sin(-oa)
    max_s = 0.0
    for i in range(N):
        vx, vy = pos[i]
        va = math.radians(ang[i])
        cva, sva = math.cos(va), math.sin(va)
        for k in range(6):
            theta = va + k * math.pi / 3
            px = vx + math.cos(theta)
            py = vy + math.sin(theta)
            # Transform to outer hexagon local frame
            dx, dy = px - ocx, py - ocy
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            # Required side for this vertex: max of 6 expressions
            s_req = max(abs(lx), abs(sqrt3 * lx + ly) / 2.0, abs(-sqrt3 * lx + ly) / 2.0)
            if s_req > max_s:
                max_s = s_req
    return max_s

def _separated(pos, ang, tol=1e-9):
    """Check all pairs of hexagons are disjoint using center distance."""
    # For two unit hexagons with centers d apart and relative rotation,
    # they are disjoint iff d >= 2 (same orientation) or slightly less (rotated).
    # Conservative: use d >= 2 * cos(min_relative_angle/2) ... 
    # Simplest safe check: d >= 2 - small_tol for same orientation.
    # For arbitrary rotations, the minimum distance for non-overlap of two unit hexagons
    # is 2 (when aligned) and can be as low as sqrt(3) (when one is rotated 30 deg).
    # Use: d >= 2 * max(cos(abs(ang_i - ang_j)/2), cos(pi/3)) = 2 * max(cos(...), 0.5)
    # Actually the inradius is sqrt(3)/2, so two hexagons touch when center distance = 2*inradius = sqrt(3)
    # if oriented optimally, or 2 (diameter) if aligned.
    # Safe universal lower bound: sqrt(3) ≈ 1.732
    # But to be safe, check actual vertex overlap for close pairs only.
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1.73:  # definitely might overlap, do detailed check
                v1 = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
                v2 = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                # SAT check
                axes = []
                for v in [v1, v2]:
                    for m in range(6):
                        e = v[(m+1)%6] - v[m]
                        nrm = math.sqrt(e[0]**2 + e[1]**2)
                        if nrm > 1e-12:
                            axes.append([e[1]/nrm, -e[0]/nrm])
                for ax in axes:
                    p1 = v1 @ ax
                    p2 = v2 @ ax
                    if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                        break
                    else:
                        continue
                else:
                    return False
            elif d < 2.0:
                # Possibly overlap with rotation, do quick check
                v1 = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
                v2 = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                axes = []
                for v in [v1, v2]:
                    for m in range(6):
                        e = v[(m+1)%6] - v[m]
                        nrm = math.sqrt(e[0]**2 + e[1]**2)
                        if nrm > 1e-12:
                            axes.append([e[1]/nrm, -e[0]/nrm])
                overlap = True
                for ax in axes:
                    p1 = v1 @ ax
                    p2 = v2 @ ax
                    if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                        overlap = False
                        break
                if overlap:
                    return False
    return True

def _hex_lattice_seed():
    """Generate hexagonal close-packed seed."""
    sqrt3 = math.sqrt(3)
    positions = []
    angles = []
    # Center + 6 neighbors + 6 next-ring (partial) = 13, take 11
    # Ring 0: center (0,0)
    positions.append([0.0, 0.0])
    angles.append(0.0)
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        theta = k * math.pi / 3
        positions.append([2.0 * math.cos(theta), 2.0 * math.sin(theta)])
        angles.append(0.0)
    # Ring 2: need 4 more from the 12 positions
    # Ring 2 positions: distance 2*sqrt(3) in directions offset by 30 deg, or 4 in cardinal
    # Actually for hex grid, ring 2 has 12 positions. Let's add 4 closest to center.
    ring2 = []
    for k in range(12):
        theta = k * math.pi / 6 + math.pi / 6  # offset by 30 deg
        # In hex grid, ring 2 vertices are at:
        # (4, 0), (2, 2*sqrt3), (-2, 2*sqrt3), (-4, 0), (-2, -2*sqrt3), (2, -2*sqrt3)
        # and the "edge midpoints" of ring 1... 
        # Let me just use the standard hex grid coordinates.
        pass
    # Simpler: use axial coordinates
    # Axial (q, r) -> cartesian: x = 2*q + r, y = sqrt(3)*r
    # Center: (0,0)
    # Ring 1: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # Ring 2: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,1)
    axial_ring1 = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    axial_ring2 = [(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,1)]
    
    positions = [[0.0, 0.0]]
    angles = [0.0]
    for q, r in axial_ring1:
        x = 2.0 * q + r
        y = sqrt3 * r
        positions.append([x, y])
        angles.append(0.0)
    # Add 4 from ring 2 (the ones closest to existing to fill gaps)
    for q, r in axial_ring2[:4]:
        x = 2.0 * q + r
        y = sqrt3 * r
        positions.append([x, y])
        angles.append(0.0)
    
    return np.array(positions[:N]), np.array(angles[:N])

def construct():
    global START_TIME, DEADLINE
    random.seed(42)
    np.random.seed(42)
    
    pos, ang = _hex_lattice_seed()
    n = N
    ocx, ocy = 0.0, 0.0
    oang = 0.0
    
    # Center the packing
    cx_mean = pos[:, 0].mean()
    cy_mean = pos[:, 1].mean()
    pos[:, 0] -= cx_mean
    pos[:, 1] -= cy_mean
    
    best_side = _required_side(pos, ang, ocx, ocy, oang)
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_ocx, best_ocy, best_oang = ocx, ocy, oang
    
    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling = 0.999
    
    while time.time() < DEADLINE - 5.0 and T > T_min:
        # Random perturbation
        new_pos = best_pos.copy()
        new_ang = best_ang.copy()
        new_ocx = best_ocx
        new_ocy = best_ocy
        new_oang = best_oang
        
        # Perturb 1-3 random hexagons
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            i = random.randint(0, n - 1)
            new_pos[i][0] += random.gauss(0, T * 0.5)
            new_pos[i][1] += random.gauss(0, T * 0.5)
            new_ang[i] += random.gauss(0, T * 10.0)
        
        # Occasionally perturb outer hexagon
        if random.random() < 0.1:
            new_ocx += random.gauss(0, T * 0.2)
            new_ocy += random.gauss(0, T * 0.2)
            new_oang += random.gauss(0, T * 5.0)
            # Keep oang in [0, 30) due to 60-degree symmetry
            new_oang = new_oang % 30.0
        
        # Check separation
        if not _separated(new_pos, new_ang):
            T *= cooling
            continue
        
        # Compute required side
        new_side = _required_side(new_pos, new_ang, new_ocx, new_ocy, new_oang)
        
        # Accept if better or with probability exp(-delta/T)
        if new_side < best_side or random.random() < math.exp(-(new_side - best_side) / T):
            best_pos = new_pos
            best_ang = new_ang
            best_ocx = new_ocx
            best_ocy = new_ocy
            best_oang = new_oang
            best_side = new_side
        
        T *= cooling
    
    # Final local refinement with smaller steps
    step = 0.05
    for _ in range(200):
        if time.time() > DEADLINE - 5.0:
            break
        improved = False
        for i in range(n):
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    trial_pos = best_pos.copy()
                    trial_pos[i][0] += dx
                    trial_pos[i][1] += dy
                    if not _separated(trial_pos, best_ang):
                        continue
                    trial_side = _required_side(trial_pos, best_ang, best_ocx, best_ocy, best_oang)
                    if trial_side < best_side - 1e-9:
                        best_pos = trial_pos
                        best_side = trial_side
                        improved = True
        if not improved:
            step *= 0.7
            if step < 0.001:
                break
    
    # Final verification
    inner = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    
    # Verify all inside outer hexagon
    for x, y, a in inner:
        verts = _hex_vertices(x, y, 1.0, a)
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], best_ocx, best_ocy, best_side, best_oang):
                # Fallback: use a larger safe side
                best_side = _required_side(best_pos, best_ang, best_ocx, best_ocy, best_oang) + 0.01
                break
    
    return inner, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oang)
