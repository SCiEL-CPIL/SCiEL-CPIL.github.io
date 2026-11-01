"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 130.0  # 20s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _soft_min_obj_and_grad(flat_pts, beta):
    """
    Soft-min objective: -log-sum-exp of (2*beta*(r_target - d_ij/2)) terms.
    We want to maximize the minimum radius. Using soft-min of "violations":
    f = -min_i g_i where g_i = min(d_ij/2, 1-|c_i|)
    
    Approximate: minimize sum of exp(-beta * h_i) where h_i are the "gaps"
    Actually, let's use: minimize -log(sum(exp(-beta * gap_i)))
    where gap_i = min over all pairs and boundary of (distance/2 or 1-|c|)
    
    Simpler: minimize the soft-min of all pairwise distances/2 and boundary distances.
    soft_min_beta(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    
    We want to maximize soft_min of (d_ij/2 for all i<j, and 1-|c_i| for all i).
    So we minimize -soft_min = (1/beta) * log(sum(exp(-beta * x_i)))
    """
    pts = flat_pts.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10, np.zeros(2 * N)
    
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Boundary distances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - norms
    
    # Collect all "gaps" (we want to maximize the minimum)
    # Pair gaps: d_ij / 2
    iu = np.triu_indices(N, k=1)
    pair_gaps = dist_mat[iu] / 2.0
    # Boundary gaps
    all_gaps = np.concatenate([pair_gaps, boundary_dists])
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * gap)))
    # For numerical stability, use log-sum-exp trick
    max_gap = np.max(all_gaps)
    if max_gap < -1000 / beta:
        # All gaps are very negative, soft min is approximately max_gap
        return -max_gap, np.zeros(2 * N)
    
    exp_terms = np.exp(-beta * (all_gaps - max_gap))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = max_gap - log_sum / beta
    
    obj = -soft_min_val  # We minimize the negative of soft-min
    
    # Gradient of obj w.r.t. pts
    # d(soft_min)/d(gap_k) = exp(-beta * gap_k) / sum(exp(-beta * gap_j))
    weights = exp_terms / np.sum(exp_terms)
    
    grad_gaps = -weights  # d(obj)/d(gap_k) since obj = -soft_min
    
    # d(gap_pair)/d(pts) for pair (i,j): gap = d_ij/2
    # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
    # d(gap_pair)/d(x_i) = (x_i - x_j) / (2 * d_ij)
    # d(gap_pair)/d(x_j) = (x_j - x_i) / (2 * d_ij)
    
    grad_pts = np.zeros((N, 2))
    
    # Pair contributions
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        d = dist_mat[i, j]
        if d > 1e-12:
            w = grad_gaps[k]
            # d(gap)/d(pts[i]) = (pts[i] - pts[j]) / (2*d)
            diff = (pts[i] - pts[j]) / (2.0 * d)
            grad_pts[i] += w * diff
            grad_pts[j] -= w * diff
    
    # Boundary contributions: gap_i = 1 - |c_i|
    # d(gap_i)/d(pts[i]) = -pts[i] / |c_i|
    for i in range(N):
        if norms[i] > 1e-12:
            w = grad_gaps[len(iu[0]) + i]
            grad_pts[i] += w * (-pts[i] / norms[i])
    
    grad_flat = grad_pts.flatten()
    
    # Penalty for going outside
    outside_mask = norms > 1.0
    if np.any(outside_mask):
        penalty = np.sum((norms[outside_mask] - 1.0) ** 2)
        obj += penalty
        # Gradient of penalty
        for i in np.where(outside_mask)[0]:
            if norms[i] > 1e-12:
                grad_pts[i] += 2.0 * (norms[i] - 1.0) * pts[i] / norms[i]
        grad_flat = grad_pts.flatten()
    
    return obj, grad_flat


def _soft_min_optimize(pts0, beta_start=50.0, beta_end=50000.0, n_stages=10, max_time=60.0):
    """Multi-stage soft-min annealing optimization."""
    pts = pts0.copy().astype(np.float64)
    start_time = time.time()
    
    for stage in range(n_stages):
        if time.time() - start_time > max_time:
            break
        beta = beta_start * (beta_end / beta_start) ** (stage / max(1, n_stages - 1))
        
        flat = pts.flatten()
        res = minimize(
            _soft_min_obj_and_grad,
            flat,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if res.fun < _soft_min_obj_and_grad(flat, beta)[0]:
            new_pts = res.x.reshape(N, 2)
            # Project back into disk
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
            pts = new_pts
    
    return pts


def _identify_contact_graph(pts, r):
    """
    Identify the active contact graph: pairs (i,j) with d_ij ≈ 2r,
    and boundary contacts i with |c_i| ≈ 1-r.
    """
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    
    norms = np.hypot(pts[:, 0], pts[:, 1])
    
    # Tolerance for "contact"
    tol = 0.01 * r + 1e-6
    
    # Pair contacts: d_ij ≈ 2r
    pair_contacts = set()
    iu = np.triu_indices(N, k=1)
    for i, j in zip(iu[0], iu[1]):
        if abs(dist_mat[i, j] - 2.0 * r) < tol:
            pair_contacts.add((i, j))
    
    # Boundary contacts: |c_i| ≈ 1 - r
    boundary_contacts = set()
    for i in range(N):
        if abs(norms[i] - (1.0 - r)) < tol:
            boundary_contacts.add(i)
    
    return pair_contacts, boundary_contacts


def _newton_raphson_active_set(pts0, pair_contacts, boundary_contacts, max_iter=200, tol=1e-10):
    """
    Newton-Raphson solver for the active-set system:
    For each pair contact (i,j): |c_i - c_j| = 2r
    For each boundary contact i: |c_i| = 1 - r
    Plus: we want to maximize r.
    
    The system has 2N + 1 unknowns (N positions + r) and M equations.
    For a rigid packing, M should be 2N+1 (or close).
    
    We solve:
    F(x) = 0 where x = (c_1, ..., c_N, r)
    
    Each pair contact gives 1 equation: |c_i - c_j|^2 - (2r)^2 = 0
    Each boundary contact gives 1 equation: |c_i|^2 - (1-r)^2 = 0
    
    We use Newton's method with the Jacobian.
    """
    n_contacts = len(pair_contacts) + len(boundary_contacts)
    
    if n_contacts == 0:
        return pts0, 0.0
    
    # Initial guess for r
    r = _score(pts0)
    if r <= 0:
        r = 0.1
    
    x = np.zeros(2 * N + 1)
    x[:2 * N] = pts0.flatten()
    x[2 * N] = r
    
    def F(x):
        """Residuals."""
        pts = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        residuals = []
        
        for (i, j) in sorted(pair_contacts):
            d2 = np.sum((pts[i] - pts[j]) ** 2)
            residuals.append(d2 - (2.0 * r) ** 2)
        
        for i in sorted(boundary_contacts):
            d2 = np.sum(pts[i] ** 2)
            residuals.append(d2 - (1.0 - r) ** 2)
        
        return np.array(residuals)
    
    def J(x):
        """Jacobian."""
        pts = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        J_mat = np.zeros((n_contacts, 2 * N + 1))
        
        row = 0
        for (i, j) in sorted(pair_contacts):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                # d/d(pts[i]) of |c_i - c_j|^2 = 2*(c_i - c_j)
                J_mat[row, 2 * i:2 * i + 2] = 2.0 * diff
                J_mat[row, 2 * j:2 * j + 2] = -2.0 * diff
                # d/dr of -(2r)^2 = -4r
                J_mat[row, 2 * N] = -4.0 * r
            row += 1
        
        for i in sorted(boundary_contacts):
            # d/d(pts[i]) of |c_i|^2 = 2*c_i
            J_mat[row, 2 * i:2 * i + 2] = 2.0 * pts[i]
            # d/dr of -(1-r)^2 = 2*(1-r)
            J_mat[row, 2 * N] = 2.0 * (1.0 - r)
            row += 1
        
        return J_mat
    
    # Newton iterations with damping
    for it in range(max_iter):
        f = F(x)
        if np.max(np.abs(f)) < tol:
            break
        
        j = J(x)
        
        # Solve J * dx = -f using least squares (may be overdetermined or underdetermined)
        try:
            dx, residuals, rank, sv = np.linalg.lstsq(j, -f, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # Damping: if step is too large, reduce it
        step_norm = np.linalg.norm(dx)
        if step_norm > 0.1:
            dx = dx * 0.1 / step_norm
        
        x_new = x + dx
        
        # Ensure r stays positive and reasonable
        if x_new[2 * N] <= 0 or x_new[2 * N] > 0.5:
            x_new[2 * N] = x[2 * N]
        
        # Check if new solution is better
        new_r = x_new[2 * N]
        new_pts = x_new[:2 * N].reshape(N, 2)
        
        # Project into disk
        norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = norms > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
        
        new_score = _score(new_pts)
        old_score = _score(pts0)
        
        if new_score >= old_score - 1e-12:
            pts0 = new_pts
            x = x_new
        else:
            # Try smaller step
            dx = dx * 0.1
            x_new = x + dx
            if x_new[2 * N] <= 0 or x_new[2 * N] > 0.5:
                x_new[2 * N] = x[2 * N]
            new_pts = x_new[:2 * N].reshape(N, 2)
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
            new_score = _score(new_pts)
            if new_score >= old_score - 1e-12:
                pts0 = new_pts
                x = x_new
    
    return pts0, x[2 * N]


def _force_directed_relax(pts0, n_iter=200, max_time=20.0):
    """Force-directed relaxation to improve initial configuration."""
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    start_time = time.time()
    
    step_size = 0.005
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Compute pairwise distances
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, 1e-10)
        
        # Repulsion force: 1/d^2 (capped)
        inv_d2 = 1.0 / (dist_mat ** 2 + 1e-8)
        inv_d2 = np.minimum(inv_d2, 100.0)  # Cap force
        np.fill_diagonal(inv_d2, 0.0)
        
        fx = np.sum(dx * inv_d2, axis=1)
        fy = np.sum(dy * inv_d2, axis=1)
        
        # Boundary force: push inward if close to boundary
        norms = np.hypot(pts[:, 0], pts[:, 1])
        boundary_force = np.where(norms > 0.9, (0.9 - norms) / norms, 0.0)
        fx += boundary_force * pts[:, 0]
        fy += boundary_force * pts[:, 1]
        
        # Normalize and update
        force_norms = np.hypot(fx, fy)
        force_norms[force_norms == 0] = 1.0
        fx = fx / force_norms
        fy = fy / force_norms
        
        new_pts = pts + step_size * np.column_stack((fx, fy))
        
        # Project into disk
        new_norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = new_norms > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_norms[mask, np.newaxis]
        
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            pts = new_pts  # Still update for continuity
            step_size *= 0.98
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Stage 1: Force-directed relaxation
    try:
        if time.time() < _DEADLINE - 80:
            relaxed_pts = _force_directed_relax(pts0, n_iter=300, max_time=20.0)
            relaxed_score = _score(relaxed_pts)
            if relaxed_score > best_score:
                best_pts = relaxed_pts
                best_score = relaxed_score
    except Exception:
        pass
    
    # Stage 2: Multi-stage soft-min annealing
    try:
        if time.time() < _DEADLINE - 40:
            soft_pts = _soft_min_optimize(
                best_pts,
                beta_start=50.0,
                beta_end=50000.0,
                n_stages=12,
                max_time=40.0
            )
            soft_score = _score(soft_pts)
            if soft_score > best_score:
                best_pts = soft_pts
                best_score = soft_score
    except Exception:
        pass
    
    # Stage 3: Identify contact graph and solve with Newton-Raphson
    try:
        if time.time() < _DEADLINE - 10:
            r_current = _score(best_pts)
            pair_contacts, boundary_contacts = _identify_contact_graph(best_pts, r_current)
            
            # Try multiple tolerances for contact identification
            for tol_factor in [0.5, 1.0, 2.0, 5.0]:
                tol = tol_factor * 0.01 * r_current + 1e-6
                dx = best_pts[:, 0:1] - best_pts[0:1, 0]
                dy = best_pts[:, 1:2] - best_pts[0:1, 1]
                dist_mat = np.sqrt(dx * dx + dy * dy)
                np.fill_diagonal(dist_mat, np.inf)
                norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
                
                pc = set()
                iu = np.triu_indices(N, k=1)
                for i, j in zip(iu[0], iu[1]):
                    if abs(dist_mat[i, j] - 2.0 * r_current) < tol:
                        pc.add((i, j))
                
                bc = set()
                for i in range(N):
                    if abs(norms[i] - (1.0 - r_current)) < tol:
                        bc.add(i)
                
                if len(pc) + len(bc) >= 2 * N:
                    new_pts, new_r = _newton_raphson_active_set(
                        best_pts, pc, bc, max_iter=300, tol=1e-12
                    )
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_pts = new_pts
                        best_score = new_score
    except Exception:
        pass
    
    # Stage 4: Final polish with soft-min at high beta
    try:
        if time.time() < _DEADLINE - 5:
            final_pts = _soft_min_optimize(
                best_pts,
                beta_start=10000.0,
                beta_end=100000.0,
                n_stages=5,
                max_time=10.0
            )
            final_score = _score(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
    except Exception:
        pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
