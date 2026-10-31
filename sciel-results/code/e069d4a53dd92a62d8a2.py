import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def get_areas_cartesian(cart):
    """Compute all triangle areas in Cartesian coordinates."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """Calculate the minimum area of all triangles formed by N points."""
    return np.min(get_areas_cartesian(points))

def generate_hexagonal_init():
    """
    Generate a symmetric hexagonal lattice initialization.
    For N=11, we can use a central point and two rings or a specific grid.
    A 4x4 triangular grid has 10 points. Adding the centroid of the large triangle
    or adjusting positions can yield 11.
    
    Let's try a "superposition" of a 3x3 grid (9 pts) + 2 extra?
    Or better: Use the vertices of a regular hexagon + center + 4 others?
    
    Actually, the optimal configuration for 11 points is likely close to a
    distorted hexagonal packing.
    
    Let's create a candidate based on barycentric coordinates of a fine grid
    and pick the 11 points that are most evenly spaced.
    
    Simpler deterministic approach:
    Start with a 4x4 grid (10 points) and add the center of the triangle (1/3, 1/3).
    Wait, (1/3, 1/3) is already in the 4x4 grid?
    Grid points (u,v) where u,v in {0, 1/3, 2/3, 1} and u+v<=1.
    (0,0), (1/3,0), (2/3,0), (1,0)
    (0,1/3), (1/3,1/3), (2/3,1/3)
    (0,2/3), (1/3,2/3)
    (0,1)
    Total 10.
    
    Let's perturb this grid slightly and add a point? No, N is fixed at 11.
    We need 11 points.
    
    Alternative: Use a 5x5 grid subset?
    Or just use a high-quality random seed with symmetry enforcement?
    
    Let's try a "wheel" pattern:
    1 center.
    6 points around it.
    4 points near the corners?
    
    Let's stick to a robust random initialization with multiple seeds,
    but structured to be close to a lattice.
    """
    candidates = []
    
    # Candidate 1: Perturbed 4x4 grid + one extra point (split a vertex?)
    # Actually, let's just use a very fine grid and pick the best 11 via a quick greedy?
    # Too slow.
    
    # Let's use a known good heuristic:
    # Place points on a hexagonal lattice clipped to the triangle.
    # Hex lattice basis: e1 = (1, 0), e2 = (1/2, sqrt(3)/2)
    # In barycentric, this is easier to generate by sampling.
    
    rng = np.random.default_rng(123)
    
    # Generate a dense set of random points and pick the best 11?
    # No, that's not an initialization, that's an algorithm.
    
    # Let's just return a few diverse random starts and let the optimizer work.
    # But the prompt asks for a specific strategy.
    # The "hexagonal init" should be structured.
    
    # Try: 3 points on each side (midpoints and vertices) = 6.
    # Center = 1.
    # 4 points in the interior forming a smaller triangle?
    
    # Let's use a simple deterministic pattern:
    # Barycentric coordinates:
    # (1/3, 1/3) - center
    # (0,0), (1,0), (0,1) - vertices
    # (1/2, 0), (0, 1/2), (1/2, 1/2) - midpoints of sides
    # (1/3, 0), (2/3, 0), (0, 1/3), (0, 2/3), (1/3, 2/3) - trisection points
    # Total: 1 + 3 + 3 + 5 = 12. Remove one?
    # Remove (1/2, 1/2) which is the midpoint of the side opposite to (0,0)?
    # (1/2, 1/2) is on the edge u+v=1.
    
    # Let's try this set of 11:
    pts = [
        [1/3, 1/3],
        [0, 0], [1, 0], [0, 1],
        [0.5, 0], [0, 0.5], # Midpoints of 2 sides
        [1/3, 0], [2/3, 0], [0, 1/3], [0, 2/3], [1/3, 2/3]
    ]
    bary = np.array(pts)
    bary = clamp_bary(bary)
    candidates.append(bary)
    
    # Candidate 2: Random with seed 42
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    bary = np.column_stack([u, v])
    bary = clamp_bary(bary)
    candidates.append(bary)
    
    # Candidate 3: Random with seed 100
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    bary = np.column_stack([u, v])
    bary = clamp_bary(bary)
    candidates.append(bary)
    
    # Evaluate and return best
    best_bary = None
    best_val = -np.inf
    for b in candidates:
        val = calc_min_area_cartesian(bary_to_cart(b))
        if val > best_val:
            best_val = val
            best_bary = b
    return best_bary

def soft_min_grad(bary_flat, beta):
    """
    Compute the soft-min objective and its gradient.
    Objective: -soft_min(areas, beta)
    soft_min(A, beta) = -1/beta * log(sum(exp(-beta * A_i)))
    We want to maximize min(A), so we minimize -min(A).
    Using soft-min as a smooth approximation:
    f(x) = -soft_min(A(x), beta) = 1/beta * log(sum(exp(-beta * A_i)))
    
    Gradient of f w.r.t x:
    df/dA_j = exp(-beta * A_j) / sum(exp(-beta * A_k)) = p_j (softmax weights)
    dA_i/dx = ...
    
    Area A_i = 0.5 * |det(P_j - P_i, P_k - P_i)|
    Let cross_i = det(P_j - P_i, P_k - P_i)
    A_i = 0.5 * |cross_i|
    
    To handle the absolute value smoothly, we assume points are in general position
    and the sign of cross_i is constant in the neighborhood.
    We can track the sign or just use the derivative of the signed area and multiply by sign.
    If we use soft-min of unsigned areas, the gradient of |cross| is sign(cross) * d(cross).
    
    Let's compute dA_i/dx for each point m.
    A_i = 0.5 * (x_j - x_i) * (y_k - y_i) - 0.5 * (x_k - x_i) * (y_j - y_i) ... wait, cross product.
    Cross = (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i)
    
    d(Cross)/dx_i = -(y_k - y_i) - (-(y_j - y_i)) = y_j - y_k
    d(Cross)/dy_i = -(x_k - x_i) - (-(x_j - x_i)) = x_j - x_k
    d(Cross)/dx_j = (y_k - y_i)
    d(Cross)/dy_j = -(x_k - x_i)
    d(Cross)/dx_k = -(y_j - y_i)
    d(Cross)/dy_k = (x_j - x_i)
    
    So dA_i/dx_m = 0.5 * sign(Cross_i) * d(Cross_i)/dx_m
    
    Total gradient grad_x = sum_i p_i * dA_i/dx
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    
    # Compute cross products
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    
    # Softmax weights
    # f = 1/beta * log(sum(exp(-beta * A)))
    # To avoid overflow, shift by max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    p = exp_scaled / sum_exp  # Weights for each triangle
    
    # Objective value
    obj_val = (max_scaled + np.log(sum_exp)) / beta
    
    # Gradient
    # grad = sum_i p_i * dA_i/dx
    # dA_i/dx = 0.5 * sign(cross_i) * d(cross_i)/dx
    
    grad = np.zeros((N, 2))
    
    # Vectorized gradient accumulation
    # For each triangle i, we need to add p_i * 0.5 * sign(cross_i) * d(cross_i)/dx_m to grad[m]
    
    signs = np.sign(cross)
    coeff = 0.5 * p * signs
    
    # d(cross)/dx_i = y_j - y_k
    # d(cross)/dy_i = x_j - x_k
    # d(cross)/dx_j = y_k - y_i
    # d(cross)/dy_j = -(x_k - x_i)
    # d(cross)/dx_k = -(y_j - y_i)
    # d(cross)/dy_k = x_j - x_i
    
    # We use np.add.at to accumulate gradients for each point index
    
    # Point i
    np.add.at(grad, (tri_idx[:, 0], 0), coeff * (p_j[:, 1] - p_k[:, 1]))
    np.add.at(grad, (tri_idx[:, 0], 1), coeff * (p_j[:, 0] - p_k[:, 0]))
    
    # Point j
    np.add.at(grad, (tri_idx[:, 1], 0), coeff * (p_k[:, 1] - p_i[:, 1]))
    np.add.at(grad, (tri_idx[:, 1], 1), coeff * (-(p_k[:, 0] - p_i[:, 0])))
    
    # Point k
    np.add.at(grad, (tri_idx[:, 2], 0), coeff * (-(p_j[:, 1] - p_i[:, 1])))
    np.add.at(grad, (tri_idx[:, 2], 1), coeff * (p_j[:, 0] - p_i[:, 0]))
    
    return obj_val, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = calc_min_area_cartesian(bary_to_cart(bary))
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate initial guess
    init_bary = generate_hexagonal_init()
    update_best(init_bary)

    # 2. Graduated Soft-Min Optimization
    # Start with low beta (smooth) and increase to high beta (sharp)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_bary = init_bary.copy()
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        def obj_grad(x):
            val, grad = soft_min_grad(x, beta)
            return val, grad
        
        try:
            res = minimize(
                obj_grad,
                current_bary.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 20}
            )
            if res.success:
                new_bary = clamp_bary(res.x.reshape(N, 2))
                # Check if it's an improvement in the hard sense
                hard_val = calc_min_area_cartesian(bary_to_cart(new_bary))
                if hard_val > best_val:
                    update_best(new_bary)
                    current_bary = new_bary
                else:
                    # Even if hard val didn't improve, the soft val might have led to a better basin
                    # But we should be careful not to diverge.
                    # Let's accept it if it's close, or just keep the best hard one.
                    # For now, keep the best hard one as the anchor.
                    pass
        except Exception:
            pass

    # 3. Final Polish with Nelder-Mead on the hard min
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: -calc_min_area_cartesian(bary_to_cart(clamp_bary(x.reshape(N, 2)))),
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            polished_bary = clamp_bary(res.x.reshape(N, 2))
            update_best(polished_bary)
        except Exception:
            pass

    # 4. Random Restarts if time permits
    if time.time() < deadline - 30:
        for seed in [7, 42, 123, 999]:
            if time.time() > deadline - 15:
                break
            rng = np.random.default_rng(seed)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            rand_bary = np.column_stack([u, v])
            rand_bary = clamp_bary(rand_bary)
            
            # Quick soft-min polish
            for beta in [50.0, 200.0]:
                if time.time() > deadline - 10:
                    break
                def obj_grad(x):
                    val, grad = soft_min_grad(x, beta)
                    return val, grad
                try:
                    res = minimize(
                        obj_grad,
                        rand_bary.flatten(),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(0, 1)] * (N * 2),
                        options={'maxiter': 20, 'ftol': 1e-9, 'gtol': 1e-6}
                    )
                    if res.success:
                        rand_bary = clamp_bary(res.x.reshape(N, 2))
                except:
                    pass
            
            update_best(rand_bary)

    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
