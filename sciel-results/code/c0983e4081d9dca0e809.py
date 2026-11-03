import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]
_SQRT3 = math.sqrt(3)


def _get_hex_verts(pos, ang_rad):
    """Compute vertices of a single hexagon."""
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    vx = pos[0] + _UX * c - _UY * s
    vy = pos[1] + _UX * s + _UY * c
    return np.column_stack((vx, vy))

def _get_hex_normals(ang_rad):
    """Compute edge normals of a single hexagon."""
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    nx = _ENX * c - _ENY * s
    ny = _ENX * s + _ENY * c
    return np.column_stack((nx, ny))

def _compute_overlap_and_gradient(pos, ang_rad, i, j):
    """
    Computes the penetration depth and gradient (force) between hex i and j.
    Returns (penetration, grad_i, grad_j).
    If disjoint, returns (0, 0, 0).
    """
    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return 0.0, np.zeros(2), np.zeros(2)

    v1 = _get_hex_verts(pos[i], ang_rad[i])
    v2 = _get_hex_verts(pos[j], ang_rad[j])
    n1 = _get_hex_normals(ang_rad[i])
    n2 = _get_hex_normals(ang_rad[j])

    min_gap = float('inf')
    best_normal = None
    best_dir = 0 # 1: push i away from j along normal, -1: push j away from i along normal? 
                 # Actually, let's define the gap such that positive is disjoint.
                 # Gap = min( max(p1) - min(p2), max(p2) - min(p1) )? No.
                 # Standard SAT: Project onto normal n.
                 # Interval 1: [min1, max1], Interval 2: [min2, max2].
                 # Gap = min(max1 - min2, max2 - min1).
                 # If Gap > 0, disjoint. If Gap < 0, overlap.
                 # We want to maximize the gap (make it least negative).
                 # The gradient of the gap w.r.t positions gives the direction to move.

    # We check normals from both hexagons
    all_normals = np.vstack([n1, n2])
    
    for k in range(12):
        nx, ny = all_normals[k]
        p1 = v1[:, 0] * nx + v1[:, 1] * ny
        p2 = v2[:, 0] * nx + v2[:, 1] * ny
        
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        
        # Gap is the distance between the intervals.
        # If they overlap, the gap is negative.
        # Gap = min(max1 - min2, max2 - min1)
        gap1 = max1 - min2
        gap2 = max2 - min1
        gap = min(gap1, gap2)
        
        if gap < min_gap:
            min_gap = gap
            best_normal = np.array([nx, ny])
            # Determine direction of push
            # If gap1 <= gap2, it means max1 < min2 is the limiting factor? 
            # No, if gap1 is small, max1 is close to min2.
            # To increase gap, we want to increase max1 (push 1 in +n) or decrease min2 (push 2 in -n).
            # So push 1 in +n, push 2 in -n.
            if gap1 <= gap2:
                best_dir = 1
            else:
                # max2 < min1. To increase gap, increase max2 (push 2 in +n) or decrease min1 (push 1 in -n).
                # So push 2 in +n, push 1 in -n.
                best_dir = -1

    if min_gap > 1e-9:
        return 0.0, np.zeros(2), np.zeros(2)
        
    # Overlap
    penetration = -min_gap
    # Gradient of penetration w.r.t pos_i and pos_j
    # We want to minimize penetration.
    # If best_dir == 1: Move i in +n, j in -n.
    # Grad_i = n, Grad_j = -n
    # If best_dir == -1: Move i in -n, j in +n.
    # Grad_i = -n, Grad_j = n
    
    if best_dir == 1:
        grad_i = best_normal
        grad_j = -best_normal
    else:
        grad_i = -best_normal
        grad_j = best_normal
        
    return penetration, grad_i, grad_j


def _compute_outer_side_and_grads(pos, ang_rad):
    """
    Computes the minimum outer hexagon side length and the gradients of the constraint 
    that the outer hexagon must contain all inner hexagons.
    Returns (side_length, grads_pos, grads_ang).
    """
    # Compute all vertices
    all_verts = np.zeros((N * 6, 2))
    for i in range(N):
        v = _get_hex_verts(pos[i], ang_rad[i])
        all_verts[i*6:(i+1)*6] = v
    
    # Center is centroid of all vertices (approximation for optimization, exact for final eval)
    center = np.mean(all_verts, axis=0)
    offsets = all_verts - center
    
    # Function to evaluate side length for a given rotation phi
    def eval_side(phi_rad):
        normal_angles = phi_rad + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0] * nx + offsets[:, 1] * ny
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)
    
    # Find optimal phi
    # Coarse scan
    phi_vals = np.radians(np.arange(0, 60, 2))
    best_phi = 0
    best_side = float('inf')
    for p in phi_vals:
        s = eval_side(p)
        if s < best_side:
            best_side = s
            best_phi = p
            
    # Fine tune with scipy
    try:
        res = minimize(eval_side, best_phi, bounds=[best_phi - 2.0, best_phi + 2.0], method='bounded', options={'xatol': 1e-8})
        if res.success:
            best_side = res.fun
            best_phi = res.x
    except:
        pass
        
    # Compute gradients
    # The side length is determined by the vertex that projects furthest onto one of the 6 normals.
    # Let k be the normal index that determines the max projection.
    # Let v_idx be the vertex index that determines the max projection for that normal.
    # side = (2/sqrt(3)) * |offset[v_idx] . normal[k]|
    # Since side depends on offset, and offset depends on pos and center, and center depends on pos.
    # This is complex. For a gradient step, we can approximate the force.
    # Force on vertex v_idx is along normal[k] (to pull it in) or -normal[k] (to push it out? No, we want to shrink side).
    # To shrink side, we want to reduce the projection.
    # If offset . normal > 0, we want to decrease it. Move vertex towards center (opposite to normal).
    # If offset . normal < 0, we want to increase it (make less negative). Move vertex towards center (along normal).
    # So force on vertex is -sign(offset . normal) * normal.
    
    grads_pos = np.zeros((N, 2))
    grads_ang = np.zeros(N)
    
    normal_angles = best_phi + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    # Find the critical vertex for each normal
    proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
    # proj shape (72, 6)
    
    # For each normal, find the vertex with max absolute projection
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0) # (6,)
    max_side = (2.0 / _SQRT3) * np.max(max_projs)
    
    # Find which normal(s) are active
    active_normals = np.where(np.isclose(max_projs, np.max(max_projs)))[0]
    
    for k in active_normals:
        # Find vertex with max abs projection for this normal
        v_idx = np.argmax(abs_proj[:, k])
        sign = np.sign(proj[v_idx, k])
        
        # Force on this vertex
        # We want to reduce the projection.
        # Gradient of projection w.r.t vertex pos is normal.
        # So we move vertex in direction -sign * normal.
        force = -sign * np.array([nx[k], ny[k]])
        
        # Distribute force to the hexagon
        h_idx = v_idx // 6
        local_v_idx = v_idx % 6
        
        # Force on center of mass of hexagon
        grads_pos[h_idx] += force
        
        # Torque on hexagon
        # Torque = r x F
        # r is vector from hex center to vertex
        r = _get_hex_verts(np.array([0,0]), ang_rad[h_idx])[local_v_idx]
        torque = r[0] * force[1] - r[1] * force[0]
        grads_ang[h_idx] += torque
        
    # Normalize gradients to have similar scale
    # Side length is ~5-10. Penetration is ~0-1.
    # We scale the outer constraint gradient to match the overlap constraint scale.
    # A typical overlap penetration is 0.1. A side change of 0.1 is significant.
    # So scale factor 1.0 is probably okay, but let's use 0.5 for outer to be safe.
    grads_pos *= 0.5
    grads_ang *= 0.1 # Angles are in radians, so torque needs scaling
    
    return float(max_side), grads_pos, grads_ang


def _is_valid(pos, ang_rad):
    for i in range(N):
        for j in range(i + 1, N):
            pen, _, _ = _compute_overlap_and_gradient(pos, ang_rad, i, j)
            if pen > 1e-9:
                return False
    return True

def _compute_min_outer_side_final(pos, ang_rad):
    """High precision calculation for final score."""
    all_verts = np.zeros((N * 6, 2))
    for i in range(N):
        v = _get_hex_verts(pos[i], ang_rad[i])
        all_verts[i*6:(i+1)*6] = v
    
    center = np.mean(all_verts, axis=0)
    offsets = all_verts - center
    
    def eval_phi(phi_rad):
        normal_angles = phi_rad + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0] * nx + offsets[:, 1] * ny
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)
    
    phi_vals = np.radians(np.arange(0, 60, 1))
    best_phi = 0
    best_side = float('inf')
    for p in phi_vals:
        s = eval_phi(p)
        if s < best_side:
            best_side = s
            best_phi = p
            
    try:
        res = minimize(eval_phi, best_phi, bounds=[best_phi - 1.0, best_phi + 1.0], method='bounded', options={'xatol': 1e-12, 'maxiter': 200})
        if res.success:
            best_side = res.fun
            best_phi = res.x
    except:
        pass
        
    return float(best_side), float(best_phi), float(center[0]), float(center[1])


def construct():
    t0 = time.time()
    
    # Initial configurations
    a1 = np.array([_SQRT3, 0.0])
    a2 = np.array([_SQRT3 / 2.0, 1.5])
    
    def make_honeycomb():
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
            ang[idx] = 0.0
        return pos, ang

    def make_random():
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(-30, 30, N)
        return pos, ang

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    inits = [
        make_honeycomb(),
        make_random(),
        make_random(),
    ]

    for init_idx, (pos0, ang0_deg) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 15.0:
            break
            
        ang0 = np.radians(ang0_deg)
        
        # Ensure initial validity via random jitter if needed
        if not _is_valid(pos0, ang0):
            for _ in range(50):
                if time.time() - t0 > DEADLINE - 15.0:
                    break
                pos0 = pos0 + np.random.uniform(-0.1, 0.1, (N, 2))
                if _is_valid(pos0, ang0):
                    break
            if not _is_valid(pos0, ang0):
                continue

        # Evaluate initial
        s0, phi0, cx0, cy0 = _compute_min_outer_side_final(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        # Gradient Descent with Subgradient Projection
        pos = pos0.copy()
        ang = ang0.copy()
        cur_side, _, _, _ = _compute_min_outer_side_final(pos, ang)
        
        # Parameters
        lr_pos = 0.05
        lr_ang = 0.05
        decay = 0.999
        steps = 0
        max_steps = 5000
        time_per_init = (DEADLINE - 20.0) / max(1, len(inits) - init_idx)
        t_init_start = time.time()

        while time.time() - t_init_start < time_per_init and steps < max_steps:
            steps += 1
            
            # Compute total gradient
            grad_pos = np.zeros((N, 2))
            grad_ang = np.zeros(N)
            
            # Overlap gradients
            for i in range(N):
                for j in range(i + 1, N):
                    pen, gi, gj = _compute_overlap_and_gradient(pos, ang, i, j)
                    if pen > 0:
                        # Weight by penetration to focus on deep overlaps
                        weight = pen
                        grad_pos[i] += weight * gi
                        grad_pos[j] += weight * gj
                        
            # Outer boundary gradients
            _, gp, ga = _compute_outer_side_and_grads(pos, ang)
            grad_pos += gp
            grad_ang += ga
            
            # Update positions
            # Use simple gradient step with decay
            pos -= lr_pos * grad_pos
            ang -= lr_ang * grad_ang
            
            # Clamp angles to avoid weird behavior (optional)
            # ang = np.clip(ang, -np.pi, np.pi)
            
            # Check validity (if we violated constraints, project back or just accept and let next step fix?)
            # For subgradient method on non-smooth functions, we often just step and evaluate.
            # However, if we create large overlaps, we might get stuck.
            # Let's add a simple repair step if overlaps are large.
            max_pen = 0
            for i in range(N):
                for j in range(i + 1, N):
                    p, _, _ = _compute_overlap_and_gradient(pos, ang, i, j)
                    if p > max_pen: max_pen = p
                    
            if max_pen > 0.5:
                # Emergency repair: push apart
                for _ in range(10):
                    moved = False
                    for i in range(N):
                        for j in range(i + 1, N):
                            p, gi, gj = _compute_overlap_and_gradient(pos, ang, i, j)
                            if p > 1e-6:
                                pos[i] += 0.1 * gi
                                pos[j] += 0.1 * gj
                                moved = True
                    if not moved: break
            
            # Evaluate
            new_side, _, _, _ = _compute_min_outer_side_final(pos, ang)
            
            if new_side < best_side:
                best_side = new_side
                best_positions = pos.copy()
                best_angles = ang.copy()
                # best_phi, best_cx, best_cy will be recomputed at the end for the best config
                
            cur_side = new_side
            lr_pos *= decay
            lr_ang *= decay
            
            if lr_pos < 1e-5:
                break

    # Final polish on the best found configuration
    if best_positions is not None:
        pos = best_positions.copy()
        ang = best_angles.copy()
        
        # Local search / Coordinate descent polish
        for _ in range(50):
            if time.time() - t0 > DEADLINE - 5.0:
                break
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = pos.copy()
                        new_ang = ang.copy()
                        step = 0.05
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step * 0.1 # Smaller angle step

                        if _is_valid(new_pos, new_ang):
                            new_side, _, _, _ = _compute_min_outer_side_final(new_pos, new_ang)
                            if new_side < best_side - 1e-10:
                                pos = new_pos
                                ang = new_ang
                                best_side = new_side
                                best_positions = pos.copy()
                                best_angles = ang.copy()
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    # Final verification and calculation
    if not _is_valid(best_positions, best_angles):
        # Fallback
        pos0, ang0_deg = make_honeycomb()
        ang0 = np.radians(ang0_deg)
        best_positions = pos0
        best_angles = ang0
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side_final(best_positions, best_angles)
    else:
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side_final(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(np.degrees(best_angles[i]))])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
