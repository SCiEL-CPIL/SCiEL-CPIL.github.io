"""Program for spectral graph lifts task."""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    edge_list = list(edges)
    
    # Import check_lift inside to avoid potential import issues at module load if path is tricky,
    # though usually it's fine. The seed did it inside.
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
    except ImportError:
        # Fallback if import fails, though task says it's available
        def check_lift(graph, signs):
            raise Exception("check_lift not available")

    def get_continuous_radius_and_grad(thetas):
        # Build continuous signed adjacency matrix M
        # M[u, v] = cos(theta_u - theta_v)
        M = np.zeros((n, n))
        for u, v in edge_list:
            val = np.cos(thetas[u] - thetas[v])
            M[u, v] = val
            M[v, u] = val
        
        # Compute eigenvalues and eigenvectors
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(M)
        lambda_max = eigvals[-1]
        x = eigvecs[:, -1]
        
        # Gradient of lambda_max with respect to M
        # d(lambda_max)/d(M_ij) = x_i x_j
        grad_M = np.outer(x, x)
        
        # Gradient with respect to thetas
        # M_uv = cos(theta_u - theta_v)
        # d(M_uv)/d(theta_k) = -sin(theta_u - theta_v) * (delta_uk - delta_vk)
        
        grad_theta = np.zeros(n)
        for u, v in edge_list:
            sin_val = np.sin(thetas[u] - thetas[v])
            # Contribution to grad_theta[u]
            # d(M_uv)/d(theta_u) = -sin(theta_u - theta_v)
            # d(M_vu)/d(theta_u) = 0 (since M_vu depends on theta_v - theta_u, d/dtheta_u is sin(theta_v - theta_u) = -sin(theta_u - theta_v) ? 
            # Wait.
            # M_vu = cos(theta_v - theta_u) = cos(theta_u - theta_v) = M_uv.
            # d(M_vu)/d(theta_u) = -sin(theta_v - theta_u) * d(theta_v - theta_u)/d(theta_u)
            # = -sin(theta_v - theta_u) * (-1) = sin(theta_v - theta_u) = -sin(theta_u - theta_v).
            
            # So both M_uv and M_vu contribute to grad_theta[u].
            # grad_theta[u] += grad_M[u, v] * (-sin_val) + grad_M[v, u] * (-sin_val)
            # Since M is symmetric, grad_M[u, v] == grad_M[v, u].
            # grad_theta[u] += 2 * grad_M[u, v] * (-sin_val)
            
            # Similarly for v:
            # d(M_uv)/d(theta_v) = -sin(theta_u - theta_v) * (-1) = sin_val
            # d(M_vu)/d(theta_v) = -sin(theta_v - theta_u) * (1) = sin(theta_u - theta_v) = sin_val
            # grad_theta[v] += 2 * grad_M[u, v] * (sin_val)
            
            grad_theta[u] += 2.0 * grad_M[u, v] * (-sin_val)
            grad_theta[v] += 2.0 * grad_M[u, v] * (sin_val)
            
        return lambda_max, grad_theta

    def objective(thetas):
        val, _ = get_continuous_radius_and_grad(thetas)
        return val

    def grad(thetas):
        _, g = get_continuous_radius_and_grad(thetas)
        return g

    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Phase optimization with random restarts
    num_restarts = 10
    for _ in range(num_restarts):
        thetas_init = rng.uniform(0, 2 * np.pi, n)
        
        try:
            res = minimize(objective, thetas_init, method='L-BFGS-B', jac=grad, options={'maxiter': 500, 'ftol': 1e-8})
            thetas_opt = res.x
            
            # Convert continuous phases to discrete signs
            signs = np.ones(len(edge_list), dtype=int)
            for i, (u, v) in enumerate(edge_list):
                val = np.cos(thetas_opt[u] - thetas_opt[v])
                # If val is very close to 0, the sign is ambiguous. 
                # Default to 1, or use a small epsilon.
                if val >= 0:
                    signs[i] = 1
                else:
                    signs[i] = -1
            
            # Validate the discrete lift
            try:
                radius, details = check_lift(graph, signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
            except Exception:
                pass # Invalid (disconnected) lift
                
        except Exception:
            pass

    # Fallback: Random search if phase optimization didn't find a valid/good solution
    if best_signs is None or best_radius > 3.0: # 3.0 is the max possible radius for 3-regular
        for _ in range(budget):
            signs = rng.choice([-1, 1], size=len(edge_list))
            try:
                radius, details = check_lift(graph, signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
            except Exception:
                pass

    if best_signs is None:
        best_signs = np.ones(len(edge_list), dtype=int)

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
