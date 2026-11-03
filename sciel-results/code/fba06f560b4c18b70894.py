"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # We optimize w (width), and for each circle: x, y, r.
    # h = 2 - w.
    # Total variables: 1 (w) + 3*N (x,y,r)
    
    def unpack(v):
        w = v[0]
        h = 2.0 - w
        xs = v[1:1+N]
        ys = v[1+N:1+2*N]
        rs = v[1+2*N:1+3*N]
        return w, h, xs, ys, rs

    def objective(v):
        w, h, xs, ys, rs = unpack(v)
        return -np.sum(rs)

    # Constraints for SLSQP
    def make_constraints():
        cons = []
        # w bounds
        cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.1}) # w > 0.1
        cons.append({'type': 'ineq', 'fun': lambda v: 1.9 - v[0]}) # w < 1.9
        
        # For each circle i:
        for i in range(N):
            idx_x = 1 + i
            idx_y = 1 + N + i
            idx_r = 1 + 2*N + i
            
            # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+i] - v[1+2*N+i]})
            # x_i <= w - r_i  =>  w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+i] - v[1+2*N+i]})
            
            # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+N+i] - v[1+2*N+i]})
            # y_i <= h - r_i = (2-w) - r_i  =>  2 - w - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[0] - v[1+N+i] - v[1+2*N+i]})
            
            # r_i > 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+2*N+i]})

        # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                idx_xi = 1 + i
                idx_yi = 1 + N + i
                idx_ri = 1 + 2*N + i
                idx_xj = 1 + j
                idx_yj = 1 + N + j
                idx_rj = 1 + 2*N + j
                
                def non_overlap(v, i=i, j=j):
                    xi = v[idx_xi]
                    yi = v[idx_yi]
                    ri = v[idx_ri]
                    xj = v[idx_xj]
                    yj = v[idx_yj]
                    rj = v[idx_rj]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                
                cons.append({'type': 'ineq', 'fun': non_overlap})
        
        return cons

    constraints = make_constraints()
    
    # Initial guess
    # Try a square aspect ratio first, w=1, h=1
    w0 = 1.0
    h0 = 1.0
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / side))
    cols = side
    
    v0 = np.zeros(1 + 3*N)
    v0[0] = w0
    
    for i in range(N):
        row = i // cols
        col = i % cols
        x = (col + 0.5) * w0 / cols
        y = (row + 0.5) * h0 / rows
        r = min(w0/cols, h0/rows) * 0.4
        
        v0[1+i] = x
        v0[1+N+i] = y
        v0[1+2*N+i] = r

    # Run SLSQP
    # SLSQP can be slow if constraints are tight, but N=21 is small.
    # We might need to try multiple initializations or use COBYLA if SLSQP fails.
    # Let's try SLSQP first.
    
    res = minimize(objective, v0, method='SLSQP', constraints=constraints, 
                   options={'maxiter': 500, 'ftol': 1e-8})
    
    if res.success:
        w, h, xs, ys, rs = unpack(res.x)
        return np.column_stack([xs, ys, rs])
    
    # Fallback if SLSQP fails: try a different initial aspect ratio or method
    # Let's try a few different initial aspect ratios with SLSQP
    
    best_res = None
    best_val = -np.inf
    
    for w_init in [0.8, 1.0, 1.2, 1.5]:
        h_init = 2.0 - w_init
        v0 = np.zeros(1 + 3*N)
        v0[0] = w_init
        
        side = int(np.ceil(np.sqrt(N * w_init / h_init)))
        if side < 2: side = 2
        rows = int(np.ceil(N / side))
        cols = side
        
        for i in range(N):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * w_init / cols
            y = (row + 0.5) * h_init / rows
            r = min(w_init/cols, h_init/rows) * 0.4
            
            v0[1+i] = x
            v0[1+N+i] = y
            v0[1+2*N+i] = r

        res = minimize(objective, v0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 300, 'ftol': 1e-7})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_res = res
    
    if best_res is not None:
        w, h, xs, ys, rs = unpack(best_res.x)
        return np.column_stack([xs, ys, rs])
    
    # Final fallback: simple grid in 2x1 rectangle
    # This might not be optimal but is valid
    w = 1.0
    h = 1.0
    xs = []
    ys = []
    rs = []
    side = int(np.ceil(np.sqrt(N)))
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) * w / side
        y = (row + 0.5) * h / side
        r = min(w/side, h/side) * 0.4
        xs.append(x)
        ys.append(y)
        rs.append(r)
        
    return np.column_stack([np.array(xs), np.array(ys), np.array(rs)])

# EVOLVE-BLOCK-END
