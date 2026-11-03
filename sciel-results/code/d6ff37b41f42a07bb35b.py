# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 195.0

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0

    # Grid for integration
    N = 10000
    x = np.linspace(-15.0, 15.0, N)
    dx = x[1] - x[0]

    # Precompute Hermite polynomials
    H0 = eval_hermite(0, x)
    H3 = eval_hermite(3, x)
    H4 = eval_hermite(4, x)
    H7 = eval_hermite(7, x)
    H8 = eval_hermite(8, x)
    H11 = eval_hermite(11, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)

    # Define basis functions G0, G4, G8 such that g = c0*G0 + c4*G4 + c8*G8
    # Constraint g(0)=0 is built in.
    # G0 = (H0 - (1/9240)H12) * gauss
    # G4 = (H4 - (3/9240)H12) * gauss
    # G8 = (H8 - (105/9240)H12) * gauss
    
    G0 = (H0 - (1.0 / h12_0) * H12) * gauss
    G4 = (H4 - (h4_0 / h12_0) * H12) * gauss
    G8 = (H8 - (h8_0 / h12_0) * H12) * gauss

    # Analytical derivatives: (Hn * exp(-x^2/2))' = 2n H_{n-1} * exp(-x^2/2)
    # G0' = (0 - (1/9240)*24*H11) * gauss = -(24/9240) H11 * gauss
    # G4' = (8*H3 - (3/9240)*24*H11) * gauss = (8*H3 - (72/9240)*H11) * gauss
    # G8' = (16*H7 - (105/9240)*24*H11) * gauss = (16*H7 - (2520/9240)*H11) * gauss
    
    G0_p = (-(24.0 / h12_0) * H11) * gauss
    G4_p = (8.0 * H3 - (72.0 / h12_0) * H11) * gauss
    G8_p = (16.0 * H7 - (2520.0 / h12_0) * H11) * gauss

    # Compute matrices M_g, M_x, M_d
    # M[i,j] = integral( G_i * G_j ) dx
    # We use the identity: integral(A+B)^2 = integral(A^2) + integral(B^2) + 2 integral(AB)
    # So integral(AB) = (integral(A+B)^2 - integral(A^2) - integral(B^2)) / 2
    
    Gs = [G0, G4, G8]
    Gp = [G0_p, G4_p, G8_p]

    M_g = np.zeros((3,3))
    M_x = np.zeros((3,3))
    M_d = np.zeros((3,3))

    # Diagonal terms
    for i in range(3):
        M_g[i,i] = np.sum(Gs[i]**2) * dx
        M_x[i,i] = np.sum((x * Gs[i])**2) * dx
        M_d[i,i] = np.sum(Gp[i]**2) * dx

    # Off-diagonal terms
    for i in range(3):
        for j in range(i+1, 3):
            G_sum = Gs[i] + Gs[j]
            Gp_sum = Gp[i] + Gp[j]
            
            Q_g_sum = np.sum(G_sum**2) * dx
            Q_x_sum = np.sum((x * G_sum)**2) * dx
            Q_d_sum = np.sum(Gp_sum**2) * dx
            
            M_g[i,j] = (Q_g_sum - M_g[i,i] - M_g[j,j]) / 2.0
            M_x[i,j] = (Q_x_sum - M_x[i,i] - M_x[j,j]) / 2.0
            M_d[i,j] = (Q_d_sum - M_d[i,i] - M_d[j,j]) / 2.0
            
            M_g[j,i] = M_g[i,j]
            M_x[j,i] = M_x[i,j]
            M_d[j,i] = M_d[i,j]

    def compute_bound(c):
        c = np.array(c)
        q_g = c @ M_g @ c
        if q_g < 1e-18:
            return 1e10
        q_x = c @ M_x @ c
        q_d = c @ M_d @ c
        
        # Uncertainty product squared: (Delta x)^2 (Delta xi)^2
        # Delta x^2 = q_x / q_g
        # Delta xi^2 = q_d / q_g
        # Product = q_x * q_d / q_g^2
        return (q_x * q_d) / (q_g * q_g)

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    seed = np.array([c0_init, c4_init, c8_init])
    
    best_score = compute_bound(seed)
    best_coeffs = seed.copy()

    def time_ok(margin=20.0):
        return (time.time() - start_time) < (deadline - margin)

    # Optimization
    if time_ok(10.0):
        rng = np.random.default_rng(42)
        starts = [seed.copy()]
        # Add more diverse starts to help escape local minima
        for _ in range(50):
            # Scale perturbations based on seed magnitude
            pert = seed + rng.normal(0, 0.5, 3)
            starts.append(pert)
            
        for s in starts:
            if not time_ok(5.0):
                break
            try:
                # Use Nelder-Mead for robustness
                res = minimize(
                    compute_bound,
                    s,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
            except Exception:
                pass

    return best_coeffs

# EVOLVE-BLOCK-END
