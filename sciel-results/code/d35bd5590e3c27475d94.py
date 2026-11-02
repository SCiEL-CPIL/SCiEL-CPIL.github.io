"""Circle packing in perimeter-4 rectangle: pack 21 circles to maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 150.0  # seconds
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def objective(x):
    """Negative sum of radii (we minimize negative)."""
    n = len(x)
    # x layout: [w, h, x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    w = x[0]
    h = x[1]
    radii = x[4::3]
    return -np.sum(radii)

def constraints(x):
    n = len(x)
    w = x[0]
    h = x[1]
    cons = []
    
    # Perimeter constraint: 2*(w+h) <= 4, i.e., w+h <= 2
    cons.append(2.0 - (w + h))
    
    # w > 0, h > 0 (small margin)
    cons.append(w - 0.01)
    cons.append(h - 0.01)
    
    # Each circle: center in rectangle, radius > 0, and non-overlap
    for i in range(N):
        xi = x[2 + 3*i]
        yi = x[3 + 3*i]
        ri = x[4 + 3*i]
        
        # Circle inside rectangle: ri <= xi, ri <= w-xi, ri <= yi, ri <= h-yi
        cons.append(xi - ri - 1e-9)
        cons.append(w - xi - ri - 1e-9)
        cons.append(yi - ri - 1e-9)
        cons.append(h - yi - ri - 1e-9)
        cons.append(ri - 1e-9)
    
    # Non-overlapping: distance between centers >= sum of radii
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = x[2+3*i], x[3+3*i], x[4+3*i]
            xj, yj, rj = x[2+3*j], x[3+3*j], x[4+3*j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            cons.append(dist - ri - rj - 1e-9)
    
    return cons

def constraint_dict(x):
    """Return constraints in dictionary format for SLSQP."""
    n = len(x)
    w = x[0]
    h = x[1]
    cons = []
    
    # Perimeter
    cons.append({'type': 'ineq', 'fun': lambda x, w=w: 2.0 - (x[0] + x[1])})
    
    # w, h bounds
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.01})
    cons.append({'type': 'ineq', 'fun': lambda x: x[1] - 0.01})
    
    # For each circle
    for i in range(N):
        base = 2 + 3*i
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[b] - x[b+2] - 1e-9})       # x - r
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[0] - x[b] - x[b+2] - 1e-9})  # w-x-r
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[b+1] - x[b+2] - 1e-9})       # y - r
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[1] - x[b+1] - x[b+2] - 1e-9})  # h-y-r
        cons.append({'type': 'ineq', 'fun': lambda x, b=base: x[b+2] - 1e-9})                 # r > 0
    
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            bi = 2 + 3*i
            bj = 2 + 3*j
            cons.append({'type': 'ineq', 'fun': lambda x, bi=bi, bj=bj: 
                         np.sqrt((x[bi]-x[bj])**2 + (x[bi+1]-x[bj+1])**2) - x[bi+2] - x[bj+2] - 1e-9})
    
    return cons

def make_initial_guess(w, h):
    """Create initial guess with circles in a grid pattern."""
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(2 + 3*N)
    x[0] = w
    x[1] = h
    
    for i in range(N):
        col = i % cols
        row = i // cols
        xi = (col + 0.5) * w / cols
        yi = (row + 0.5) * h / rows
        ri = min(w / cols, h / rows) * 0.45
        x[2 + 3*i] = xi
        x[3 + 3*i] = yi
        x[4 + 3*i] = ri
    
    return x

def construct():
    best_result = None
    best_value = -np.inf
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.25, 0.6, 1.8]
    
    for ar in aspect_ratios:
        if time_remaining() < 15:
            break
        
        # w + h = 2, so w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        x0 = make_initial_guess(w, h)
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=constraint_dict(x0),
                bounds=[(0.1, 1.9)] + [(0.01, 1.99)]*2 + [(0.005, 1.0)]*N + [(0.005, 1.0)]*N + [(0.005, 1.0)]*N,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < best_value:
                # Verify and extract
                w_res = res.x[0]
                h_res = res.x[1]
                circles = []
                valid = True
                for i in range(N):
                    xi = res.x[2+3*i]
                    yi = res.x[3+3*i]
                    ri = res.x[4+3*i]
                    if ri <= 0:
                        valid = False
                        break
                    circles.append([xi, yi, ri])
                
                if valid:
                    # Check constraints
                    if (w_res + h_res <= 2.0 + 1e-6 and 
                        all(0 <= c[0] and c[0] <= w_res and 0 <= c[1] and c[1] <= h_res for c in circles)):
                        # Check non-overlap
                        overlap = False
                        for i in range(N):
                            for j in range(i+1, N):
                                dist = np.sqrt((circles[i][0]-circles[j][0])**2 + (circles[i][1]-circles[j][1])**2)
                                if dist < circles[i][2] + circles[j][2] - 1e-6:
                                    overlap = True
                                    break
                            if overlap:
                                break
                        if not overlap:
                            val = sum(c[2] for c in circles)
                            if val > best_value:
                                best_value = val
                                best_result = np.array(circles)
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
