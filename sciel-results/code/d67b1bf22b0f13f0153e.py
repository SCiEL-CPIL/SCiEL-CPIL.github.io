"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _compute_score(x):
    """Compute sum of radii for a valid packing. x = [x1..xn, y1..yn, r1..rn, w]"""
    n = N
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    w = x[3*n]
    
    # Perimeter constraint: 2*(w + h) <= 4, so h = 2 - w
    h = 2.0 - w
    
    # Check bounds
    if w <= 0 or h <= 0:
        return 0.0
    if np.any(rs <= 0):
        return 0.0
    if np.any(xs - rs < -1e-9) or np.any(xs + rs > w + 1e-9):
        return 0.0
    if np.any(ys - rs < -1e-9) or np.any(ys + rs > h + 1e-9):
        return 0.0
    
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return 0.0
    
    return np.sum(rs)

def _objective(x):
    """Negative sum of radii for minimization."""
    return -_compute_score(x)

def _constraints(x):
    """Constraints for SLSQP."""
    n = N
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    w = x[3*n]
    h = 2.0 - w
    
    cons = []
    
    # Width and height bounds
    cons.append({'type': 'ineq', 'fun': lambda x: x[3*n] - 0.1})
    cons.append({'type': 'ineq', 'fun': lambda x: (2.0 - x[3*n]) - 0.1})
    
    # Each circle within bounds
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*n] - x[i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (2.0 - x[3*n]) - x[n+i] - x[2*n+i]})
    
    # Pairwise disjointness
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                         (x[i]-x[j])**2 + (x[n+i]-x[n+j])**2 - (x[2*n+i]+x[2*n+j])**2})
    
    return cons

def _grid_init(w, h):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    for idx in range(N):
        i = idx % side
        j = idx // side
        xs[idx] = (i + 0.5) * w / side
        ys[idx] = (j + 0.5) * h / side
        rs[idx] = 0.45 * min(w, h) / side
    
    return xs, ys, rs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_x = None
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for w_frac in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * w_frac / (1.0 + w_frac)
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1:
            continue
        
        xs, ys, rs = _grid_init(w, h)
        x0 = np.concatenate([xs, ys, rs, [w]])
        
        # Check initial validity
        score0 = _compute_score(x0)
        if score0 > best_score:
            best_score = score0
            best_x = x0.copy()
        
        # Run SLSQP optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=[(0.01, 1.99)] * (3*N) + [(0.1, 1.9)],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_x = result.x.copy()
        except Exception:
            pass
        
        # If time is running out, do a quick polish with the best so far
        if time.time() > deadline - 20 and best_x is not None:
            break
    
    # Final polish pass if we have time
    if best_x is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_x,
                method='SLSQP',
                bounds=[(0.01, 1.99)] * (3*N) + [(0.1, 1.9)],
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Construct the output
    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    xs = best_x[:N]
    ys = best_x[N:2*N]
    rs = best_x[2*N:3*N]
    
    # Ensure all radii are positive and circles are within bounds
    w = best_x[3*N]
    h = 2.0 - w
    
    # Clamp positions
    xs = np.clip(xs, rs, w - rs)
    ys = np.clip(ys, rs, h - rs)
    
    # Verify final result
    final_score = _compute_score(np.concatenate([xs, ys, rs, [w]]))
    if final_score <= 0:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([xs, ys, rs])

# EVOLVE-BLOCK-END
