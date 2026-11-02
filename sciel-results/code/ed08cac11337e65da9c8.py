"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Precompute all triangle indices
    tri_indices = list(itertools.combinations(range(N), 3))
    n_tris = len(tri_indices)
    # Convert to arrays for vectorization
    idx_i = np.array([t[0] for t in tri_indices])
    idx_j = np.array([t[1] for t in tri_indices])
    idx_k = np.array([t[2] for t in tri_indices])

    def bary_to_cartesian(bary):
        """Convert barycentric coords (l1, l2) to Cartesian in the unit equilateral triangle."""
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        x = 0.5 - 0.5 * l1 + 0.5 * l2
        y = (1 - l1 - l2) * np.sqrt(3) / 2
        return np.column_stack([x, y])

    def get_valid_bary(params):
        """Reshape and clip parameters to valid barycentric coordinates."""
        bary = params.reshape(N, 2)
        bary = np.clip(bary, 1e-6, 1.0 - 1e-6)
        s = bary[:, 0] + bary[:, 1]
        # If sum > 1 - eps, scale down
        mask = s > 1.0 - 1e-6
        if np.any(mask):
            bary[mask, 0] *= (1.0 - 1e-6) / s[mask]
            bary[mask, 1] *= (1.0 - 1e-6) / s[mask]
        return bary

    def compute_obj_grad(params, temperature):
        """Compute soft-min objective and gradient vectorized."""
        bary = get_valid_bary(params)
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        l3 = 1.0 - l1 - l2
        
        # Construct 3x3 matrices for all triangles
        # M[m, 0, :] = bary[i, :]
        # M[m, 1, :] = bary[j, :]
        # M[m, 2, :] = bary[k, :]
        M = np.zeros((n_tris, 3, 3))
        M[:, 0, :] = bary[idx_i, :]
        M[:, 1, :] = bary[idx_j, :]
        M[:, 2, :] = bary[idx_k, :]
        
        # Compute determinants
        # np.linalg.det works on stacks of matrices
        dets = np.linalg.det(M)
        
        # Areas are absolute determinants
        areas = np.abs(dets)
        
        # Avoid log(0)
        areas = np.maximum(areas, 1e-12)
        
        # Soft min: -t * log(sum(exp(-area/t)))
        # Stable version:
        max_neg = np.max(-areas / temperature)
        exp_terms = np.exp(-areas / temperature - max_neg)
        sum_exp = np.sum(exp_terms)
        log_sum_exp = max_neg + np.log(sum_exp)
        
        obj_val = -temperature * log_sum_exp
        
        # Weights for gradient
        weights = exp_terms / sum_exp
        
        # Gradient calculation
        # d(obj)/d(params) = sum_m weights[m] * sign(det[m]) * d(det[m])/d(params)
        # d(det)/d(l_{i,1}) = C_{0,0} - C_{0,2}
        # d(det)/d(l_{i,2}) = C_{0,1} - C_{0,2}
        # Where C is the cofactor matrix.
        # adj(M)_{c, r} = C_{r, c}.
        # So we need adj[:, 0, 0] - adj[:, 2, 0] for dl1
        # and adj[:, 1, 0] - adj[:, 2, 0] for dl2
        
        # Compute adjugate matrices
        # inv(M) = adj(M) / det(M)
        # adj(M) = inv(M) * det(M)
        # Handle singular matrices (det=0) by setting adj to 0 or using pseudo-inverse?
        # If det is 0, gradient is undefined, but in practice areas > 0.
        # We can use np.linalg.inv and multiply by det.
        # To be safe against singularities, we can add a small epsilon to det for inversion?
        # No, better to just compute cofactors directly or use a robust method.
        # Given time constraints, np.linalg.inv might fail if singular.
        # Let's try np.linalg.inv. If it fails, we fall back to finite differences or skip.
        # But for optimization, we assume non-singular.
        
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # Fallback: return large gradient to push away? Or zero?
            # Zero gradient will stop progress.
            # Let's return zeros and let optimizer handle it or use finite diff if needed.
            # But this should be rare.
            return obj_val, np.zeros(N * 2)

        adj_M = inv_M * dets[:, None, None]
        
        # C_{0,0} is adj_M[:, 0, 0]
        # C_{0,2} is adj_M[:, 2, 0]
        # C_{0,1} is adj_M[:, 1, 0]
        
        grad_det_l1_i = adj_M[:, 0, 0] - adj_M[:, 2, 0]
        grad_det_l2_i = adj_M[:, 1, 0] - adj_M[:, 2, 0]
        
        # Same for j (row 1) and k (row 2)
        # C_{1,0} is adj_M[:, 0, 1]
        # C_{1,2} is adj_M[:, 2, 1]
        # C_{1,1} is adj_M[:, 1, 1]
        
        grad_det_l1_j = adj_M[:, 0, 1] - adj_M[:, 2, 1]
        grad_det_l2_j = adj_M[:, 1, 1] - adj_M[:, 2, 1]
        
        # C_{2,0} is adj_M[:, 0, 2]
        # C_{2,2} is adj_M[:, 2, 2]
        # C_{2,1} is adj_M[:, 1, 2]
        
        grad_det_l1_k = adj_M[:, 0, 2] - adj_M[:, 2, 2]
        grad_det_l2_k = adj_M[:, 1, 2] - adj_M[:, 2, 2]
        
        # Sign of determinant
        signs = np.sign(dets)
        
        # Weighted sum
        w = weights * signs
        
        grad_bary = np.zeros((N, 2))
        
        # Accumulate gradients for i, j, k
        # np.add.at for scattered addition
        np.add.at(grad_bary, (idx_i, 0), w * grad_det_l1_i)
        np.add.at(grad_bary, (idx_i, 1), w * grad_det_l2_i)
        np.add.at(grad_bary, (idx_j, 0), w * grad_det_l1_j)
        np.add.at(grad_bary, (idx_j, 1), w * grad_det_l2_j)
        np.add.at(grad_bary, (idx_k, 0), w * grad_det_l1_k)
        np.add.at(grad_bary, (idx_k, 1), w * grad_det_l2_k)
        
        # The objective is -soft_min, so we minimize -soft_min?
        # No, we want to MAXIMIZE min_area.
        # So we MINIMIZE -min_area.
        # Soft min approximates min_area.
        # So we minimize -soft_min.
        # The gradient of -soft_min is -grad(soft_min).
        # Wait, my obj_val is -soft_min?
        # Let's check:
        # soft_min = -t * log(sum(exp(-A/t)))
        # We want to maximize min(A).
        # So we minimize -min(A).
        # Approximation: minimize -soft_min(A).
        # So objective function returns -soft_min.
        # Gradient of objective = - gradient(soft_min).
        # My computed grad_bary is gradient of soft_min w.r.t bary?
        # d(soft_min)/d(bary) = sum w * d(A)/d(bary).
        # So grad_obj = -grad_bary.
        
        grad_obj = -grad_bary
        return obj_val, grad_obj.flatten()

    def symmetric_init():
        """Create a symmetric initial configuration for 11 points."""
        bary = np.zeros((N, 2))
        
        # Center
        bary[0] = [1/3, 1/3]
        
        # 3 points near midpoints of sides (slightly inside)
        # Side 1 (l1=0): midpoint l2=0.5
        bary[1] = [0.05, 0.5]
        # Side 2 (l2=0): midpoint l1=0.5
        bary[2] = [0.5, 0.05]
        # Side 3 (l3=0): midpoint l1=0.5, l2=0.5
        bary[3] = [0.5, 0.5]
        
        # 6 points in a ring around center
        # In barycentric, "ring" is tricky. Let's use Cartesian-like projection or just place them symmetrically.
        # A regular hexagon in Cartesian maps to a distorted hexagon in Barycentric.
        # Let's place them at specific barycentric coordinates that form a hexagon.
        # Vertices of a hexagon centered at (1/3, 1/3).
        # We can define them by rotating a vector in the barycentric plane.
        # The metric in barycentric coords is not Euclidean.
        # However, for initialization, a rough hexagon is fine.
        r = 0.15
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            # Simple circular placement in l1-l2 plane
            bary[4+i, 0] = 1/3 + r * np.cos(angles[i])
            bary[4+i, 1] = 1/3 + r * np.sin(angles[i])
            
        # Ensure validity
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        
        return bary

    def random_init(seed):
        """Create a random initial configuration."""
        np.random.seed(seed)
        # Generate random points in barycentric simplex
        r = np.random.exponential(1.0, (N, 3))
        r /= r.sum(axis=1, keepdims=True)
        bary = r[:, :2]
        
        # Add small noise
        bary += np.random.normal(0, 0.01, (N, 2))
        
        # Clip to valid range
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        
        return bary

    best_result = None
    best_score = -np.inf

    # Initializations
    inits = [
        symmetric_init(),
        random_init(1),
        random_init(2),
        random_init(3),
        random_init(4),
        random_init(5),
    ]

    for init_bary in inits:
        # Annealing schedule
        temps = [0.05, 0.02, 0.01, 0.005, 0.001]
        current_bary = init_bary.copy()
        
        for temp in temps:
            try:
                # Define objective and gradient for this temperature
                def obj(params):
                    val, _ = compute_obj_grad(params, temp)
                    return val
                
                def grad(params):
                    _, g = compute_obj_grad(params, temp)
                    return g

                result = minimize(
                    obj,
                    current_bary.flatten(),
                    method='L-BFGS-B',
                    jac=grad,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score: # Note: result.fun is -min_area, so lower is better for minimization
                    # But we track best_score as max min_area
                    final_bary = get_valid_bary(result.x)
                    # Compute true min area for comparison
                    l1 = final_bary[:, 0]
                    l2 = final_bary[:, 1]
                    l3 = 1.0 - l1 - l2
                    bary_full = np.column_stack([l1, l2, l3])
                    
                    # Calculate min area
                    min_area = np.inf
                    # Vectorized min area calculation
                    M = np.zeros((n_tris, 3, 3))
                    M[:, 0, :] = bary_full[idx_i, :]
                    M[:, 1, :] = bary_full[idx_j, :]
                    M[:, 2, :] = bary_full[idx_k, :]
                    dets = np.linalg.det(M)
                    areas = np.abs(dets)
                    true_min = np.min(areas)
                    
                    if true_min > best_score:
                        best_score = true_min
                        best_result = final_bary.copy()
                        current_bary = final_bary.copy()
                        
            except Exception as e:
                # Continue with next temp or init
                pass

    if best_result is None:
        best_result = symmetric_init()

    final_bary = best_result
    cart = bary_to_cartesian(final_bary)
    
    # Ensure points are strictly inside
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
