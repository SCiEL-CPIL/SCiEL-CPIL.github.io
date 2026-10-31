"""Circle packing in perimeter-4 rectangle: Physics-based optimization with smooth penalties."""
import numpy as np
import time

N = 21
_DEADLINE = 145.0
_START = None


def _rect_dims(w):
    return w, 2.0 - w


def _grid_seed(w, h):
    """Place N circles in a grid, trying multiple aspect-friendly layouts."""
    best_seed = None
    best_score = -1e9
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.999
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                x[idx] = r
                x[N + idx] = cx
                x[2 * N + idx] = cy
                idx += 1
        if _is_feasible(x, w, tol=1e-8):
            score = np.sum(x[:N])
            if score > best_score:
                best_score = score
                best_seed = x.copy()
    if best_seed is None:
        best_seed = np.zeros(3 * N)
        best_seed[:N] = 1e-4
        best_seed[N:2 * N] = w / 2.0
        best_seed[2 * N:3 * N] = h / 2.0
    return best_seed


def _is_feasible(x, w, tol=1e-6):
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol) or np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(x):
    return float(np.sum(x[:N]))


def _physics_optimize(seed, w, h, max_iters=500, lr=0.05, stiffness=10.0):
    """
    Optimize circle positions and radii using a physics-based approach.
    We maximize the sum of radii by applying forces that expand circles
    while repelling overlapping circles and keeping them within bounds.
    """
    x = seed.copy()
    best_x = x.copy()
    best_val = -1e9
    
    for it in range(max_iters):
        if time.time() - _START > _DEADLINE - 10:
            break
            
        r = x[:N]
        cx = x[N:2*N]
        cy = x[2*N:3*N]
        
        # Gradients for radii: want to increase r, but penalize overlaps/bounds
        grad_r = np.ones(N)
        grad_cx = np.zeros(N)
        grad_cy = np.zeros(N)
        
        # Pairwise repulsion forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d2 = dx*dx + dy*dy
                min_d = r[i] + r[j]
                if d2 < min_d*min_d and d2 > 1e-12:
                    d = np.sqrt(d2)
                    # Force is proportional to overlap
                    overlap = min_d - d
                    # Normalize direction
                    nx = dx / d
                    ny = dy / d
                    
                    # Force on positions
                    # We want to push apart, so gradient of penalty is towards separation
                    # Penalty ~ overlap^2. Grad penalty wrt cx[i] is -overlap * nx
                    # We minimize penalty, so we move in direction -grad_penalty
                    # But we are maximizing sum(r) - penalty.
                    # Let's define objective J = sum(r) - stiffness * sum(overlaps^2)
                    # dJ/dcxi = -2 * stiffness * overlap * (cx_i - cx_j)/d
                    # So gradient descent step: cx_i += lr * dJ/dcxi
                    
                    force_mag = 2 * stiffness * overlap
                    grad_cx[i] -= force_mag * nx
                    grad_cy[i] -= force_mag * ny
                    grad_cx[j] += force_mag * nx
                    grad_cy[j] += force_mag * ny
                    
                    # Gradient wrt radii: dJ/dri = 1 - 2 * stiffness * overlap * (1)
                    # Because d(overlap)/dri = 1
                    grad_r[i] -= 2 * stiffness * overlap
                    grad_r[j] -= 2 * stiffness * overlap
                elif d2 <= 1e-12:
                    # Random nudge
                    angle = np.random.uniform(0, 2*np.pi)
                    nx = np.cos(angle)
                    ny = np.sin(angle)
                    force_mag = 2 * stiffness * min_d
                    grad_cx[i] -= force_mag * nx
                    grad_cy[i] -= force_mag * ny
                    grad_cx[j] += force_mag * nx
                    grad_cy[j] += force_mag * ny
                    grad_r[i] -= 2 * stiffness * min_d
                    grad_r[j] -= 2 * stiffness * min_d

        # Boundary constraints: keep circles inside [0, w] x [0, h]
        # Penalty for being out of bounds or too close to bounds
        # We want cx_i >= r_i and cx_i <= w - r_i
        # Let's use soft constraints:
        # If cx_i < r_i, penalty increases as we go left.
        # If cx_i > w - r_i, penalty increases as we go right.
        
        for i in range(N):
            # Left bound
            if cx[i] < r[i]:
                overlap = r[i] - cx[i]
                grad_cx[i] -= 2 * stiffness * overlap
                grad_r[i] -= 2 * stiffness * overlap # d(cx_i < r_i)/dr_i = -1? No, penalty is (r_i - cx_i)^2. d/dr_i = 2(r_i-cx_i). So grad_r gets -2*stiffness*overlap? 
                # Wait, J = ... - stiffness * (r_i - cx_i)^2. 
                # dJ/dr_i = 1 - 2*stiffness*(r_i - cx_i). 
                # So grad_r[i] -= 2 * stiffness * overlap is correct for the penalty part.
                # But we want to minimize penalty, so we move in direction of gradient of J.
                # If cx_i < r_i, we want to move cx_i right (increase) and r_i down (decrease) or cx_i right.
                # dJ/dcxi = -2*stiffness*(r_i - cx_i) * (-1) = 2*stiffness*(r_i-cx_i). 
                # So grad_cx[i] should be POSITIVE to move cx_i right?
                # Let's re-evaluate.
                # Penalty P = (r_i - cx_i)^2. 
                # dP/dcxi = -2(r_i - cx_i). 
                # We want to minimize P, so we do cx_i -= lr * dP/dcxi = cx_i += lr * 2(r_i-cx_i).
                # So grad_cx[i] (as in x += lr * grad) should be +2*stiffness*overlap.
                # My previous code for pairwise: 
                # overlap = min_d - d. P = overlap^2. 
                # dP/dcxi = 2*overlap * d(overlap)/dcxi = 2*overlap * (cx_i-cx_j)/d = 2*overlap*nx.
                # Minimize P -> cx_i -= lr * 2*overlap*nx.
                # So grad_cx[i] should be -2*stiffness*overlap*nx. This matches my pairwise code.
                
                # So for bounds:
                # P_left = (r_i - cx_i)^2. dP/dcxi = -2(r_i-cx_i). Minimize -> cx_i += 2(r_i-cx_i).
                # So grad_cx[i] += 2 * stiffness * overlap.
                grad_cx[i] += 2 * stiffness * overlap 
                
                # dP/dr_i = 2(r_i - cx_i). Minimize -> r_i -= 2(r_i-cx_i).
                # So grad_r[i] -= 2 * stiffness * overlap. (This matches)
            
            # Right bound
            if cx[i] > w - r[i]:
                overlap = cx[i] - (w - r[i])
                # P = (cx_i - (w-r_i))^2. dP/dcxi = 2(cx_i - w + r_i).
                # Minimize -> cx_i -= 2(cx_i - w + r_i).
                # So grad_cx[i] -= 2 * stiffness * overlap.
                grad_cx[i] -= 2 * stiffness * overlap
                
                # dP/dr_i = -2(cx_i - w + r_i). 
                # Minimize -> r_i += 2(cx_i - w + r_i).
                # So grad_r[i] += 2 * stiffness * overlap.
                grad_r[i] += 2 * stiffness * overlap

            # Bottom bound
            if cy[i] < r[i]:
                overlap = r[i] - cy[i]
                grad_cy[i] += 2 * stiffness * overlap
                grad_r[i] -= 2 * stiffness * overlap

            # Top bound
            if cy[i] > h - r[i]:
                overlap = cy[i] - (h - r[i])
                grad_cy[i] -= 2 * stiffness * overlap
                grad_r[i] += 2 * stiffness * overlap

        # Update positions and radii
        # Learning rate decay
        current_lr = lr * (1.0 - it / max_iters)
        
        x[:N] += current_lr * grad_r
        x[N:2*N] += current_lr * grad_cx
        x[2*N:3*N] += current_lr * grad_cy
        
        # Clip radii to be positive
        x[:N] = np.maximum(x[:N], 1e-4)
        
        # Check feasibility and update best
        if _is_feasible(x, w, tol=1e-7):
            val = _sum_radii(x)
            if val > best_val:
                best_val = val
                best_x = x.copy()
                
    return best_x


def construct():
    global _START
    if _START is None:
        _START = time.time()

    best_result = None
    best_val = -1e9

    # Wider set of aspect ratios (w, with h = 2 - w)
    aspect_ratios = [
        1.0, 1.1, 0.9, 1.2, 0.833, 1.3, 0.769, 1.4, 0.714,
        1.5, 0.667, 1.6, 0.625, 1.8, 0.556, 1.05, 0.95,
        1.15, 0.87, 1.25, 0.8
    ]

    for w in aspect_ratios:
        if time.time() - _START > _DEADLINE - 15:
            break
        if w <= 0.1 or w >= 1.9:
            continue
        h = 2.0 - w
        seed = _grid_seed(w, h)

        # Run physics optimization
        x_opt = _physics_optimize(seed, w, h, max_iters=800, lr=0.02, stiffness=50.0)
        x_opt[:N] = np.maximum(x_opt[:N], 1e-4)

        # Check feasibility of the polished result
        if _is_feasible(x_opt, w, tol=1e-7):
            val = _sum_radii(x_opt)
            if val > best_val:
                best_val = val
                best_result = (x_opt.copy(), w)

    # If nothing feasible, return a trivial valid packing
    if best_result is None:
        w = 1.0
        h = 1.0
        r = 1e-4
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        return np.column_stack((xs, ys, r))

    x, w = best_result
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]

    # Clamp to be safe
    r = np.maximum(r, 1e-4)
    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)

    # Re-check feasibility after clamping; if broken, shrink radii
    if not _is_feasible(np.concatenate([r, cx, cy]), w, tol=1e-5):
        for scale in [0.999, 0.995, 0.99, 0.98]:
            r_s = r * scale
            cx_s = np.clip(cx, r_s, w - r_s)
            cy_s = np.clip(cy, r_s, h - r_s)
            if _is_feasible(np.concatenate([r_s, cx_s, cy_s]), w, tol=1e-5):
                r, cx, cy = r_s, cx_s, cy_s
                break

    return np.column_stack((cx, cy, r))
