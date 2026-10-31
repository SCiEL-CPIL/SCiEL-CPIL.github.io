"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3 = np.sqrt(3.0)
TRI_AREA = S3 / 4.0  # area of the equilateral triangle with side 1

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])

def bary_to_cart(b):
    """Convert barycentric coords (u,v,w with u+v+w=1) to cartesian.
    We use (u,v) with w = 1-u-v, so point = u*V0 + v*V1 + w*V2."""
    u, v = b[:, 0], b[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * (S3 / 2.0)
    return np.column_stack([x, y])

def cart_to_bary(p):
    """Convert cartesian to barycentric (u,v) where point = u*V0 + v*V1 + w*V2."""
    x, y = p[:, 0], p[:, 1]
    w = y / (S3 / 2.0)
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def min_triangle_area_bary(b):
    """Compute minimum triangle area among all triples, in barycentric space.
    Area of triangle with barycentric coords (u1,v1),(u2,v2),(u3,v3) is
    TRI_AREA * |det of the 3x3 matrix with rows [u_i, v_i, w_i]| where w_i=1-u_i-v_i."""
    n = len(b)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Barycentric coordinates
                u1, v1 = b[i, 0], b[i, 1]
                u2, v2 = b[j, 0], b[j, 1]
                u3, v3 = b[k, 0], b[k, 1]
                w1 = 1.0 - u1 - v1
                w2 = 1.0 - u2 - v2
                w3 = 1.0 - u3 - v3
                # Determinant of [[u1,v1,w1],[u2,v2,w2],[u3,v3,w3]]
                det = (u1*(v2*w3 - w2*v3) - v1*(u2*w3 - w2*u3) + w1*(u2*v3 - v2*u3))
                area = TRI_AREA * abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def objective_smooth(b_flat, soft=0.05):
    """Smooth approximation of negative min triangle area using log-sum-exp."""
    b = b_flat.reshape(-1, 2)
    n = len(b)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                u1, v1 = b[i, 0], b[i, 1]
                u2, v2 = b[j, 0], b[j, 1]
                u3, v3 = b[k, 0], b[k, 1]
                w1 = 1.0 - u1 - v1
                w2 = 1.0 - u2 - v2
                w3 = 1.0 - u3 - v3
                det = (u1*(v2*w3 - w2*v3) - v1*(u2*w3 - w2*u3) + w1*(u2*v3 - v2*u3))
                areas.append(abs(det))
    areas = np.array(areas)
    # Log-sum-exp for smooth min
    m = np.min(areas)
    log_sum_exp = m + np.log(np.sum(np.exp(areas - m)))
    return -TRI_AREA * log_sum_exp

def in_triangle(b):
    """Check if all barycentric coordinates are valid (u>=0, v>=0, u+v<=1)."""
    u, v = b[:, 0], b[:, 1]
    return np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u + v <= 1.0 + 1e-9)

def penalty_function(b_flat, penalty_weight=100.0):
    """Add penalty for points outside the triangle."""
    b = b_flat.reshape(-1, 2)
    u, v = b[:, 0], b[:, 1]
    w = 1.0 - u - v
    # Penalty for negative barycentric coords
    pen = 0.0
    for val in np.concatenate([u, v, w]):
        if val < 0:
            pen += penalty_weight * val * val
    return pen

def objective_with_penalty(b_flat, soft=0.05, penalty_weight=100.0):
    return objective_smooth(b_flat, soft) + penalty_function(b_flat, penalty_weight)

def random_bary_init(rng):
    """Generate random valid barycentric coordinates."""
    b = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.02, 0.98)
            v = rng.uniform(0.02, 0.98 - u)
            if u + v < 0.98:
                b[i] = [u, v]
                break
    return b

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    n_restarts = 50
    restarts_done = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Different initialization strategies
        if restart < 10:
            # Random uniform
            b_init = random_bary_init(np.random.default_rng(restart * 100 + 1))
        elif restart < 20:
            # Grid-based with jitter
            b_init = np.zeros((N, 2))
            # Place points in a rough grid
            positions = [
                (0.33, 0.33), (0.66, 0.33), (0.5, 0.66),
                (0.16, 0.16), (0.83, 0.16), (0.5, 0.83),
                (0.5, 0.16), (0.16, 0.5), (0.83, 0.5),
                (0.33, 0.66), (0.66, 0.66)
            ]
            for i, (u, v) in enumerate(positions):
                b_init[i] = [u + rng.uniform(-0.05, 0.05), v + rng.uniform(-0.05, 0.05)]
        elif restart < 30:
            # Fibonacci-like spiral in barycentric space
            golden = (1 + np.sqrt(5)) / 2
            b_init = np.zeros((N, 2))
            for i in range(N):
                angle = i * 2 * np.pi / golden
                r = 0.3 * np.sqrt((i + 1) / N)
                u = 0.5 + r * np.cos(angle)
                v = 0.5 / np.sqrt(3) + r * np.sin(angle) / np.sqrt(3)
                # Clamp to valid region
                u = max(0.02, min(0.98, u))
                v = max(0.02, min(0.98 - u, v))
                b_init[i] = [u, v]
        else:
            # Perturbed version of best found so far
            if best_bary is not None:
                b_init = best_bary + rng.normal(0, 0.05, (N, 2))
                # Project back to valid region
                for i in range(N):
                    u, v = b_init[i]
                    u = max(0.02, min(0.98, u))
                    v = max(0.02, min(0.98 - u, v))
                    b_init[i] = [u, v]
            else:
                b_init = random_bary_init(np.random.default_rng(restart * 100 + 1))
        
        b_flat = b_init.flatten()
        
        # Try different softness values
        soft_values = [0.02, 0.05, 0.1]
        
        for soft in soft_values:
            if time.time() > deadline - 15.0:
                break
            
            try:
                result = minimize(
                    objective_with_penalty,
                    b_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                    args=(soft, 100.0)
                )
                
                b_result = result.x.reshape(-1, 2)
                
                # Check validity
                if not in_triangle(b_result):
                    # Project to boundary
                    for i in range(N):
                        u, v = b_result[i]
                        u = max(0.0, min(1.0, u))
                        v = max(0.0, min(1.0 - u, v))
                        b_result[i] = [u, v]
                
                # Evaluate true min area
                score = min_triangle_area_bary(b_result)
                
                if score > best_score:
                    best_score = score
                    best_bary = b_result.copy()
                    
            except Exception:
                pass
        
        restarts_done += 1
    
    if best_bary is None:
        # Fallback: simple random
        best_bary = random_bary_init(np.random.default_rng(0))
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    # Final check: ensure points are strictly inside
    for i in range(N):
        x, y = cart[i]
        # Check if point is inside triangle
        if x < 0.001 or x > 0.999 or y < 0.001 or y > S3/2 - 0.001:
            # Nudge inward
            cart[i, 0] = 0.5 + 0.4 * (cart[i, 0] - 0.5)
            cart[i, 1] = 0.5 * S3/3 + 0.4 * (cart[i, 1] - 0.5 * S3/3)
    
    return cart

# Pre-compute all triangle index combinations for speed
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Pre-compute all C(11,3) = 165 triangle index triples
    triples = list(combinations(range(N), 3))
    
    def min_area_fast(b):
        """Fast min area computation using precomputed triples."""
        min_a = np.inf
        for (i, j, k) in triples:
            u1, v1 = b[i, 0], b[i, 1]
            u2, v2 = b[j, 0], b[j, 1]
            u3, v3 = b[k, 0], b[k, 1]
            w1 = 1.0 - u1 - v1
            w2 = 1.0 - u2 - v2
            w3 = 1.0 - u3 - v3
            det = (u1*(v2*w3 - w2*v3) - v1*(u2*w3 - w2*u3) + w1*(u2*v3 - v2*u3))
            a = abs(det)
            if a < min_a:
                min_a = a
        return TRI_AREA * min_a
    
    def obj_smooth(b_flat, soft=0.05):
        b = b_flat.reshape(-1, 2)
        areas = np.empty(len(triples))
        for idx, (i, j, k) in enumerate(triples):
            u1, v1 = b[i, 0], b[i, 1]
            u2, v2 = b[j, 0], b[j, 1]
            u3, v3 = b[k, 0], b[k, 1]
            w1 = 1.0 - u1 - v1
            w2 = 1.0 - u2 - v2
            w3 = 1.0 - u3 - v3
            det = (u1*(v2*w3 - w2*v3) - v1*(u2*w3 - w2*u3) + w1*(u2*v3 - v2*u3))
            areas[idx] = abs(det)
        m = np.min(areas)
        lse = m + np.log(np.sum(np.exp(areas - m)))
        return -TRI_AREA * lse
    
    def obj_pen(b_flat, soft=0.05, pw=50.0):
        b = b_flat.reshape(-1, 2)
        u, v = b[:, 0], b[:, 1]
        w = 1.0 - u - v
        pen = pw * np.sum(np.maximum(0, -u)**2 + np.maximum(0, -v)**2 + np.maximum(0, -w)**2)
        return obj_smooth(b_flat, soft) + pen
    
    best_bary = None
    best_score = -np.inf
    
    rng = np.random.default_rng(12345)
    
    n_restarts = 80
    
    for r in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Initialization
        if r < 15:
            # Uniform random
            b_init = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u = rng.uniform(0.03, 0.97)
                    v = rng.uniform(0.03, 0.97 - u)
                    b_init[i] = [u, v]
                    break
        elif r < 30:
            # Structured grid
            b_init = np.zeros((N, 2))
            grid = [
                (1/3, 1/3), (2/3, 1/3), (1/2, 1/2),
                (1/6, 1/6), (5/6, 1/6), (1/2, 5/6),
                (1/2, 1/6), (1/6, 1/2), (5/6, 1/2),
                (1/3, 2/3), (2/3, 2/3)
            ]
            for i, (gu, gv) in enumerate(grid):
                b_init[i] = [gu + rng.normal(0, 0.03), gv + rng.normal(0, 0.03)]
        elif r < 45:
            # Fibonacci spiral
            phi = (1 + np.sqrt(5)) / 2
            b_init = np.zeros((N, 2))
            for i in range(N):
                theta = i * 2 * np.pi / phi
                rad = 0.35 * np.sqrt((i + 0.5) / N)
                u = 0.5 + rad * np.cos(theta)
                v = S3/6 + rad * np.sin(theta) * 0.577
                u = np.clip(u, 0.03, 0.97)
                v = np.clip(v, 0.03, 0.97 - u)
                b_init[i] = [u, v]
        else:
            # Perturb best
            if best_bary is not None:
                b_init = best_bary + rng.normal(0, 0.04, (N, 2))
                for i in range(N):
                    u, v = b_init[i]
                    u = np.clip(u, 0.03, 0.97)
                    v = np.clip(v, 0.03, 0.97 - u)
                    b_init[i] = [u, v]
            else:
                b_init = np.zeros((N, 2))
                for i in range(N):
                    u = rng.uniform(0.03, 0.97)
                    v = rng.uniform(0.03, 0.97 - u)
                    b_init[i] = [u, v]
        
        b_flat = b_init.flatten()
        
        # Try optimization with different softness
        for soft in [0.03, 0.06, 0.12]:
            if time.time() > deadline - 20.0:
                break
            try:
                res = minimize(
                    obj_pen, b_flat, method='L-BFGS-B',
                    args=(soft, 50.0),
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                b_res = res.x.reshape(-1, 2)
                # Project to valid
                for i in range(N):
                    u, v = b_res[i]
                    u = max(0.0, min(1.0, u))
                    v = max(0.0, min(1.0 - u, v))
                    b_res[i] = [u, v]
                sc = min_area_fast(b_res)
                if sc > best_score:
                    best_score = sc
                    best_bary = b_res.copy()
            except Exception:
                pass
    
    if best_bary is None:
        best_bary = np.array([(1/3, 1/3), (2/3, 1/3), (1/2, 1/2),
                              (1/6, 1/6), (5/6, 1/6), (1/2, 5/6),
                              (1/2, 1/6), (1/6, 1/2), (5/6, 1/2),
                              (1/3, 2/3), (2/3, 2/3)], dtype=float)
    
    # Convert to cartesian
    u, v = best_bary[:, 0], best_bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * (S3 / 2.0)
    cart = np.column_stack([x, y])
    
    return cart
