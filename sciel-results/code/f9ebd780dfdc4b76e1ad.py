"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_valid(x, y, r):
    """
    Adjust radii to ensure validity (no overlaps, inside box).
    Returns modified x, y, r.
    """
    r_new = r.copy()
    for i in range(N):
        min_r = r_new[i]
        # Boundary constraints
        min_r = min(min_r, x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        # Overlap constraints
        for j in range(N):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                # r_i + r_j <= d  =>  r_i <= d - r_j
                min_r = min(min_r, d - r_new[j])
        r_new[i] = max(0.0, min_r)
    return x, y, r_new

def _optimize_once(init_x, init_y, init_r, time_budget):
    """
    Run Adam optimizer for circle packing.
    """
    x = init_x.copy()
    y = init_y.copy()
    r = init_r.copy()
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_init = 0.005
    lr = lr_init
    
    m_x = np.zeros(N)
    v_x = np.zeros(N)
    m_y = np.zeros(N)
    v_y = np.zeros(N)
    m_r = np.zeros(N)
    v_r = np.zeros(N)
    
    t = 0
    best_score = -np.inf
    best_state = (x.copy(), y.copy(), r.copy())
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    start = time.time()
    
    while time.time() - start < time_budget:
        t += 1
        # Decay learning rate
        lr = lr_init * (0.9995 ** t)
        
        # 1. Compute Distances and Penalties
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx**2 + dy**2)
        
        overlap = r[i_idx] + r[j_idx] - d
        overlap_pos = np.maximum(overlap, 0)
        
        inv_d = np.where(d > 1e-9, 1.0/d, 0.0)
        
        force_x = 2.0 * overlap_pos * inv_d * dx
        force_y = 2.0 * overlap_pos * inv_d * dy
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        np.add.at(grad_x, i_idx, force_x)
        np.add.at(grad_x, j_idx, -force_x)
        np.add.at(grad_y, i_idx, force_y)
        np.add.at(grad_y, j_idx, -force_y)
        
        bound_x_left = np.maximum(r - x, 0)
        bound_x_right = np.maximum(r - (1 - x), 0)
        bound_y_bottom = np.maximum(r - y, 0)
        bound_y_top = np.maximum(r - (1 - y), 0)
        
        grad_x += 2.0 * bound_x_left - 2.0 * bound_x_right
        grad_y += 2.0 * bound_y_bottom - 2.0 * bound_y_top
        
        grad_r = np.zeros(N)
        np.add.at(grad_r, i_idx, 2.0 * overlap_pos)
        np.add.at(grad_r, j_idx, 2.0 * overlap_pos)
        
        grad_r += 2.0 * (bound_x_left + bound_x_right + bound_y_bottom + bound_y_top)
        
        g_x = grad_x
        g_y = grad_y
        g_r = grad_r - 1.0
        
        # Adam Update
        m_x = beta1 * m_x + (1 - beta1) * g_x
        v_x = beta2 * v_x + (1 - beta2) * (g_x ** 2)
        m_x_hat = m_x / (1 - beta1 ** t)
        v_x_hat = v_x / (1 - beta2 ** t)
        x -= lr * m_x_hat / (np.sqrt(v_x_hat) + eps)
        
        m_y = beta1 * m_y + (1 - beta1) * g_y
        v_y = beta2 * v_y + (1 - beta2) * (g_y ** 2)
        m_y_hat = m_y / (1 - beta1 ** t)
        v_y_hat = v_y / (1 - beta2 ** t)
        y -= lr * m_y_hat / (np.sqrt(v_y_hat) + eps)
        
        m_r = beta1 * m_r + (1 - beta1) * g_r
        v_r = beta2 * v_r + (1 - beta2) * (g_r ** 2)
        m_r_hat = m_r / (1 - beta1 ** t)
        v_r_hat = v_r / (1 - beta2 ** t)
        r -= lr * m_r_hat / (np.sqrt(v_r_hat) + eps)
        
        r = np.maximum(r, 0.0)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        # Validity Check and Score
        # Check for overlaps
        max_overlap = np.max(overlap_pos)
        if max_overlap < 1e-6:
            # Check boundaries
            if np.all(x >= r) and np.all(x <= 1-r) and np.all(y >= r) and np.all(y <= 1-r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_state = (x.copy(), y.copy(), r.copy())
        
    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = -np.inf
    best_result = None
    
    # Generate Initializations
    inits = []
    
    # 1. Hexagonal Grid
    r_approx = 0.12
    dx = 2 * r_approx
    dy = dx * np.sqrt(3) / 2
    
    points = []
    nx = int(1 / dx) + 2
    ny = int(1 / dy) + 2
    
    for i in range(nx):
        for j in range(ny):
            px = i * dx + (j % 2) * dx / 2 + 0.5
            py = j * dy + 0.5
            if 0 <= px <= 1 and 0 <= py <= 1:
                points.append([px, py])
    
    np.random.shuffle(points)
    if len(points) >= N:
        init_pts = np.array(points[:N])
        init_pts += np.random.normal(0, 0.02, init_pts.shape)
        init_pts = np.clip(init_pts, 0.05, 0.95)
        inits.append((init_pts[:, 0], init_pts[:, 1], np.full(N, r_approx*0.8)))
    
    # 2. Random Initializations
    for _ in range(6):
        rx = np.random.uniform(0.1, 0.9, N)
        ry = np.random.uniform(0.1, 0.9, N)
        rr = np.full(N, 0.1)
        inits.append((rx, ry, rr))
    
    # 3. Grid Initialization
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.5/side, 1 - 0.5/side, side)
    ys = np.linspace(0.5/side, 1 - 0.5/side, side)
    gpts = []
    for i in range(side):
        for j in range(side):
            if len(gpts) < N:
                gpts.append([xs[i], ys[j]])
    gpts = np.array(gpts)
    gpts += np.random.normal(0, 0.01, gpts.shape)
    gpts = np.clip(gpts, 0.05, 0.95)
    inits.append((gpts[:, 0], gpts[:, 1], np.full(N, 0.1)))
    
    # Optimize each init
    for i, (ix, iy, ir) in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        time_budget = min(15.0, deadline - time.time() - 5)
        if time_budget <= 0: break
        
        bx, by, br = _optimize_once(ix, iy, ir, time_budget)
        
        # Enforce validity
        bx, by, br = _make_valid(bx, by, br)
        score = np.sum(br)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack((bx, by, br))
            
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
