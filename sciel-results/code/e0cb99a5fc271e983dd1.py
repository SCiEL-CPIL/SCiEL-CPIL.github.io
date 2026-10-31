"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
import scipy.optimize as optimize
import time

def construct():
    # Precompute constants for the Hermite basis {0, 4, 8, 12}
    Ks = np.array([0, 4, 8, 12])
    Ns = len(Ks)
    
    # Norms N_ii = integral h_i^2 = sqrt(pi) * 2^i * i!
    # h_i(x) = H_i(x) * exp(-x^2/2)
    N_diag = np.array([np.sqrt(np.pi) * (2**k) * sp.factorial(k, exact=True) for k in Ks])
    N_mat = np.diag(N_diag)
    
    # Compute M_ij = integral x^2 H_i(x) H_j(x) exp(-x^2) dx
    M_mat = np.zeros((Ns, Ns))
    for i in range(Ns):
        for j in range(i, Ns):
            def integrand(x):
                hi = sp.eval_hermite(Ks[i], x)
                hj = sp.eval_hermite(Ks[j], x)
                return x**2 * hi * hj * np.exp(-x**2)
            val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
            M_mat[i,j] = val
            M_mat[j,i] = val

    # Constraint: g(0) = 0
    # g(0) = sum c_k H_k(0) = 0
    H0_0 = sp.eval_hermite(0, 0)
    H4_0 = sp.eval_hermite(4, 0)
    H8_0 = sp.eval_hermite(8, 0)
    H12_0 = sp.eval_hermite(12, 0)
    
    # c_12 = -(c_0 H0_0 + c_4 H4_0 + c_8 H8_0) / H12_0
    # Let u=c_0, v=c_4, w=c_8
    # c = [u, v, w, c12]

    def get_c(u, v, w):
        c12 = -(u * H0_0 + v * H4_0 + w * H8_0) / H12_0
        return np.array([u, v, w, c12])

    def objective(params):
        u, v, w = params
        c = get_c(u, v, w)
        num = c @ M_mat @ c
        den = c @ N_mat @ c
        if den <= 1e-15:
            return np.inf
        # We minimize A(g) = num/den. Since A(g)=A(g_hat), product is (num/den)^2.
        # Minimizing num/den is equivalent.
        return num / den

    start_time = time.time()
    deadline = start_time + 180 # Safety margin

    best_val = np.inf
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]) # Seed
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    starts = [best_params]
    for _ in range(30):
        starts.append(rng.uniform(-10, 10, size=3))
        
    for s in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = optimize.minimize(objective, s, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # High-precision polish
    if time.time() < deadline - 5:
        try:
            res = optimize.minimize(objective, best_params, method='BFGS', options={'gtol': 1e-16, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
