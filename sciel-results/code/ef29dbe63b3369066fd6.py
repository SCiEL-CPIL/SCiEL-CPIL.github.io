import numpy as np
import time


def _rot_matrix(roll_deg, pitch_deg, yaw_deg):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in degrees."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rx @ Ry @ Rz


def _pose_points(pose, pts):
    """Apply pose (x,y,z,yaw,pitch,roll) to points: R@pts + t"""
    x, y, z, yaw, pitch, roll = pose
    R = _rot_matrix(roll, pitch, yaw)
    return R @ pts.T + np.array([x, y, z]).reshape(3, 1)


def _in_corridor(pts):
    """Check if all points are in the corridor."""
    x, y, z = pts[0], pts[1], pts[2]
    r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return np.all(r1 | r2 | r3)


def _densify_path(path, max_step=0.05, max_angle_step=2.0):
    """Insert linearly interpolated poses."""
    result = [path[0]]
    for i in range(1, len(path)):
        p0 = path[i - 1]
        p1 = path[i]
        pos_dist = np.linalg.norm(p1[:3] - p0[:3])
        ang_dists = np.abs(p1[3:] - p0[3:])
        n_pos = int(np.ceil(pos_dist / max_step)) if pos_dist > 0 else 0
        n_ang = int(np.ceil(np.max(ang_dists) / max_angle_step)) if np.max(ang_dists) > 0 else 0
        n = max(n_pos, n_ang, 1)
        for j in range(1, n + 1):
            t = j / n
            interp = p0 + t * (p1 - p0)
            result.append(interp)
    return np.array(result)


def _evaluate_path(path):
    """Evaluate the volume of the swept shape for a given path."""
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    n_pts = len(pts)
    
    start_pose = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_pose = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    
    full_path = np.vstack([start_pose, path, end_pose])
    dense_path = _densify_path(full_path)
    
    survive = np.ones(n_pts, dtype=bool)
    
    for pose in dense_path:
        placed = _pose_points(pose, pts[survive])
        still_in = _in_corridor(placed)
        survive_idx = np.where(survive)[0]
        survive[survive_idx[~still_in]] = False
        if not np.any(survive):
            break
    
    if not np.any(survive):
        return 0
    
    survive_grid = survive.reshape(81, 21, 21)
    
    count = 0
    for i in range(78):
        for j in range(18):
            for k in range(18):
                block = survive_grid[i:i+4, j:j+4, k:k+4]
                if np.all(block):
                    count += 1
    
    return count


def _generate_structured_path():
    """
    Generate a structured path that keeps the sofa centered in the corridor.
    
    Strategy:
    - Phase 1: Move from x=-3 to x=0.5 in the left corridor (y in [0,1], z in [0,1])
      Keep sofa centered at y=0.5, z=0.5 with identity rotation.
    - Phase 2: Move along the horizontal corridor from x=0.5, y=0.5 to x=0.5, y=4.5
      Keep sofa centered, identity rotation.
    - Phase 3: Corner rotation at y~5. Rotate 90 degrees about x-axis while
      moving from y=4.5 to y=5.5, keeping the sofa within the corridor.
    - Phase 4: Move up in z from z=0.5 to z=6 in the vertical corridor.
    """
    path = []
    
    # Phase 1: Enter from left, move to x=0.5, y=0.5, z=0.5
    # Start will be prepended by evaluator at (-3,0,0,0,0,0)
    # So we start our path near x=0
    n1 = 20
    for i in range(n1 + 1):
        t = i / n1
        x = -1.0 + t * 1.5  # from -1 to 0.5
        y = 0.5
        z = 0.5
        path.append([x, y, z, 0.0, 0.0, 0.0])
    
    # Phase 2: Move along horizontal corridor from y=0.5 to y=4.5
    n2 = 40
    for i in range(1, n2 + 1):
        t = i / n2
        x = 0.5
        y = 0.5 + t * 4.0  # from 0.5 to 4.5
        z = 0.5
        path.append([x, y, z, 0.0, 0.0, 0.0])
    
    # Phase 3: Corner - rotate 90 deg about x-axis
    # At the corner, we need to go from horizontal (facing +y) to vertical (facing +z)
    # Rotate about x-axis: roll goes from 0 to 90
    # During rotation, the sofa's extent in y and z changes
    # Keep x=0.5, move y from 4.5 to 5.5, z from 0.5 to 1.0
    n3 = 40
    for i in range(1, n3 + 1):
        t = i / n3
        x = 0.5
        y = 4.5 + t * 1.0  # from 4.5 to 5.5
        z = 0.5 + t * 0.5   # from 0.5 to 1.0
        roll = t * 90.0     # 0 to 90
        path.append([x, y, z, 0.0, 0.0, roll])
    
    # Phase 4: Move up in z from z=1.0 to z=6.0
    n4 = 60
    for i in range(1, n4 + 1):
        t = i / n4
        x = 0.5
        y = 5.5
        z = 1.0 + t * 5.0   # from 1.0 to 6.0
        path.append([x, y, z, 0.0, 0.0, 90.0])
    
    return np.array(path)


def moving_sofa_3d_path() -> np.ndarray:
    """
    Find a motion path for the 3D moving sofa problem that maximizes the
    swept volume. Uses a structured path design based on corridor geometry.
    """
    deadline = time.time() + 570  # safety margin
    
    # Start with structured path
    path = _generate_structured_path()
    best_score = _evaluate_path(path)
    best_path = path.copy()
    
    # Local refinement: try to improve by adjusting key parameters
    # Parameters to tune:
    # - x position in corridor (0 to 1)
    # - y,z centering
    # - rotation profile at corner
    
    # Try different x positions
    for x_pos in [0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.45, 0.55]:
        if time.time() > deadline:
            break
        test_path = best_path.copy()
        # Adjust x for all poses where x is in [0,1]
        for i in range(len(test_path)):
            if 0 <= test_path[i, 0] <= 1:
                test_path[i, 0] = x_pos
        score = _evaluate_path(test_path)
        if score > best_score:
            best_score = score
            best_path = test_path
    
    # Try different y centering in horizontal corridor
    for y_off in [0.4, 0.45, 0.5, 0.55, 0.6]:
        if time.time() > deadline:
            break
        test_path = best_path.copy()
        for i in range(len(test_path)):
            if test_path[i, 3] < 45 and test_path[i, 5] < 45:  # before rotation
                test_path[i, 1] = y_off
        score = _evaluate_path(test_path)
        if score > best_score:
            best_score = score
            best_path = test_path
    
    # Try different z centering
    for z_off in [0.4, 0.45, 0.5, 0.55, 0.6]:
        if time.time() > deadline:
            break
        test_path = best_path.copy()
        for i in range(len(test_path)):
            if test_path[i, 3] < 45 and test_path[i, 5] < 45:
                test_path[i, 2] = z_off
        score = _evaluate_path(test_path)
        if score > best_score:
            best_score = score
            best_path = test_path
    
    # Try different rotation profiles at corner
    # The corner is where roll goes from 0 to 90
    # Try different y,z trajectories during rotation
    for y_start in [4.0, 4.2, 4.4, 4.5, 4.6, 4.8]:
        if time.time() > deadline:
            break
        for y_end in [5.0, 5.2, 5.4, 5.5, 5.6, 5.8]:
            if time.time() > deadline:
                break
            for z_start in [0.3, 0.4, 0.5, 0.6]:
                if time.time() > deadline:
                    break
                for z_end in [0.7, 0.8, 0.9, 1.0]:
                    if time.time() > deadline:
                        break
                    test_path = best_path.copy()
                    # Find corner poses (where roll is between 0 and 90)
                    for i in range(len(test_path)):
                        if 0 <= test_path[i, 5] <= 90:
                            t = test_path[i, 5] / 90.0
                            test_path[i, 1] = y_start + t * (y_end - y_start)
                            test_path[i, 2] = z_start + t * (z_end - z_start)
                    score = _evaluate_path(test_path)
                    if score > best_score:
                        best_score = score
                        best_path = test_path
    
    # Final local search with small perturbations
    np.random.seed(42)
    n_iter = 0
    while time.time() < deadline - 30:
        n_iter += 1
        idx = np.random.randint(1, len(best_path) - 1)
        delta = np.random.randn(6) * 0.03
        delta[3:] *= 0.3
        
        new_path = best_path.copy()
        new_path[idx] = best_path[idx] + delta
        
        score = _evaluate_path(new_path)
        if score > best_score:
            best_score = score
            best_path = new_path
    
    return best_path
