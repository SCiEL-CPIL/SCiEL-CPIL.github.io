"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Check overlap: dist_sq >= (r_i + r_j)^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """
    Shrink radii and adjust positions to ensure strict validity.
    This is a safety net.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    for _ in range(100):
        valid = True
        
        # 1. Ensure circles are inside the square
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-6)
                valid = False
        
        # 2. Ensure no overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down both radii to just fit
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        scale = (dist * 0.9999) / total_r
                        if scale < 1.0:
                            radii[i] *= scale
                            radii[j] *= scale
                            valid = False
                    else:
                        radii[i] = 1e-6
                        radii[j] = 1e-6
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _get_contact_graph(pos, radii, eps=1e-4):
    """
    Get contact graph: pairs of circles that are close to touching.
    Returns list of (i, j) where dist(i,j) <= r_i + r_j + eps
    """
    contacts = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= radii[i] + radii[j] + eps:
                contacts.append((i, j))
    return contacts

def _solve_qp_for_radii(pos, contacts, deadline):
    """
    Solve a QP to maximize sum of radii given fixed positions and contact graph.
    Uses linearized distance constraints from the contact graph.
    
    For each contact (i,j): r_i + r_j <= dist(i,j)
    For boundaries: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    """
    n = len(radii) if 'radii' in locals() else N
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(N)
    
    # Inequality constraints: A @ r <= b
    A_rows = []
    b_vals = []
    
    # Boundary constraints
    for i in range(N):
        # r_i <= x_i
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(pos[i, 0])
        
        # r_i <= 1 - x_i
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(1.0 - pos[i, 0])
        
        # r_i <= y_i
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(pos[i, 1])
        
        # r_i <= 1 - y_i
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(1.0 - pos[i, 1])
    
    # Contact constraints: r_i + r_j <= dist(i,j)
    for i, j in contacts:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        row = np.zeros(N)
        row[i] = 1.0
        row[j] = 1.0
        A_rows.append(row)
        b_vals.append(dist)
    
    if len(A_rows) == 0:
        return None, None
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Solve LP: max c^T r s.t. A r <= b, r >= 0
    # Use scipy's linprog
    try:
        res = minimize(
            lambda r: -np.sum(r),
            np.zeros(N),
            method='SLSQP',
            bounds=[(1e-6, 0.5)] * N,
            constraints=[
                {'type': 'ineq', 'fun': lambda r, A=A, b=b: b - A @ r}
            ]
        )
        if res.success:
            return res.x, None
    except Exception:
        pass
    
    return None, None

def _graph_based_optimization(pos, radii, deadline, max_iters=50):
    """
    Graph-based optimization:
    1. Build contact graph
    2. Solve QP for radii
    3. Adjust positions to satisfy contacts
    4. Repeat
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Get contact graph
        contacts = _get_contact_graph(pos, radii)
        
        # Solve for optimal radii given positions and contacts
        new_radii, _ = _solve_qp_for_radii(pos, contacts, deadline)
        if new_radii is not None:
            radii = new_radii
        
        # Adjust positions to make contacts tight
        # For each contact, if dist > r_i + r_j, move circles closer
        for i, j in contacts:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-8:
                # Move each circle half the gap toward the other
                gap = dist - (radii[i] + radii[j])
                if gap > 1e-6:
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    # Move i toward j, j toward i
                    pos[i, 0] += nx * gap / 2
                    pos[i, 1] += ny * gap / 2
                    pos[j, 0] -= nx * gap / 2
                    pos[j, 1] -= ny * gap / 2
        
        # Clamp positions to valid range
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
        
        # Post-process to ensure validity
        pos_pp, radii_pp = _post_process(pos, radii)
        
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        pos, radii = pos_pp, radii_pp
    
    return best_pos, best_radii

def _initial_hexagonal():
    """
    Hexagonal packing initialization.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal close packing
    # Rows of circles with alternating offsets
    row_spacing = 0.2
    col_spacing = 0.2
    r = 0.09
    
    idx = 0
    for row in range(5):
        for col in range(6):
            if idx >= N:
                break
            x = 0.1 + col * col_spacing + (row % 2) * col_spacing / 2
            y = 0.1 + row * row_spacing
            if x + r <= 1.0 and y + r <= 1.0:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    
    # Fill remaining with small circles
    for i in range(idx, N):
        pos[i] = [0.5, 0.5]
        radii[i] = 0.02
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_corner_biased(seed):
    """
    Initialization with circles biased toward corners.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place some circles in corners
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for i, (cx, cy) in enumerate(corners):
        if i < N:
            pos[i] = [cx, cy]
            radii[i] = 0.12
    
    # Place remaining in center area
    for i in range(4, N):
        pos[i] = rng.uniform(0.2, 0.8, 2)
        radii[i] = 0.06
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_best_fit():
    """
    Best-fit initialization: place circles greedily in largest available space.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with 4 corner circles
    corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.10
    
    # Add center circle
    pos[4] = [0.5, 0.5]
    radii[4] = 0.14
    
    # Add more circles in gaps
    gap_positions = [
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5],
        [0.35, 0.5], [0.65, 0.5], [0.5, 0.35], [0.5, 0.65],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.4, 0.4], [0.6, 0.4], [0.4, 0.6], [0.6, 0.6],
        [0.3, 0.5], [0.7, 0.5], [0.5, 0.3], [0.5, 0.7]
    ]
    
    for i in range(5, N):
        if i - 5 < len(gap_positions):
            pos[i] = gap_positions[i - 5]
            radii[i] = 0.05
        else:
            pos[i] = [0.5, 0.5]
            radii[i] = 0.03
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.02, 0.1, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_grid():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 5x5 grid
    side = 5
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.19
    # Fill remaining if any
    for i in range(N):
        if radii[i] == 0:
            pos[i] = [0.5, 0.5]
            radii[i] = 0.01
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # List of initializations
    inits = []
    inits.append(_initial_best_fit())
    inits.append(_initial_hexagonal())
    inits.append(_initial_corner_biased(42))
    inits.append(_initial_grid())
    for i in range(3):
        inits.append(_initial_random(i * 100))
        
    candidates = []
    
    for idx, (pos, radii) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Graph-based optimization
        gb_deadline = min(deadline - 15, time.time() + 15)
        pos_gb, radii_gb = _graph_based_optimization(pos, radii, gb_deadline)
        
        # Post-process
        pos_pp, radii_pp = _post_process(pos_gb, radii_gb)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            candidates.append((score, pos_pp, radii_pp))
            
            if score > best_score:
                best_score = score
                best_pos = pos_pp
                best_radii = radii_pp
                
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # If time remains, do another round of optimization on top candidates
    if time.time() < deadline - 10:
        for score, pos_c, radii_c in candidates[:2]:
            if time.time() > deadline - 5:
                break
            
            gb_deadline = min(deadline - 3, time.time() + 8)
            pos_gb2, radii_gb2 = _graph_based_optimization(pos_c, radii_c, gb_deadline)
            
            pos_pp2, radii_pp2 = _post_process(pos_gb2, radii_gb2)
            if _is_valid(pos_pp2, radii_pp2):
                final_score = np.sum(radii_pp2)
                if final_score > best_score:
                    best_score = final_score
                    best_pos = pos_pp2
                    best_radii = radii_pp2
    
    # Final safety net
    if best_pos is None:
        best_pos, best_radii = _initial_grid()
        
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to simple grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.19
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
