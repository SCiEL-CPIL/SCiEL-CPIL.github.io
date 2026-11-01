"""Analytic Hermite-basis approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 175


def _hermite_moments():
    """
    Compute the exact inner products for the Hermite-Gaussian basis.
    
    We use the probabilists' Hermite polynomials H_n(x) defined by:
    H_0 = 1
    H_1 = x
    H_{n+1} = x H_n - n H_{n-1}
    
    The basis functions are phi_n(x) = H_n(x) * exp(-x^2/2).
    
    Key identities:
    - <phi_m, phi_n> = delta_{mn} * 2^n n! / sqrt(2*pi)  (orthogonality)
    - <phi_m, x phi_n> = sqrt((n+1)) phi_{n+1} + sqrt(n) phi_{n-1}  (recurrence)
      More precisely: x H_n = H_{n+1} + n H_{n-1}
      So x phi_n = phi_{n+1} + n phi_{n-1}
      <phi_m, x phi_n> = <phi_m, phi_{n+1}> + n <phi_m, phi_{n-1}>
                       = delta_{m,n+1} * 2^{n+1}(n+1)!/sqrt(2pi) + n * delta_{m,n-1} * 2^{n-1}(n-1)!/sqrt(2pi)
    
    For the momentum operator, in the Fourier domain:
    F(phi_n)(xi) = (-i)^n phi_n(xi)  (for the convention F(f)(xi) = (1/sqrt(2pi)) int f(x) e^{-ixxi} dx)
    
    So <G, xi^2 G> where G = F(g):
    If g = sum a_n phi_n, then G = sum a_n (-i)^n phi_n
    <G, xi^2 G> = sum_{m,n} a_m a_n (-i)^m (-i)^n <phi_m, xi^2 phi_n>
    
    And <phi_m, xi^2 phi_n> is computed the same way as <phi_m, x^2 phi_n> since the operator structure is the same.
    
    x^2 H_n = (x H_n) * x = (H_{n+1} + n H_{n-1}) * x = (H_{n+2} + (n+1)H_n) + n(H_n + (n-1)H_{n-2})
            = H_{n+2} + (2n+1) H_n + n(n-1) H_{n-2}
    
    So x^2 phi_n = phi_{n+2} + (2n+1) phi_n + n(n-1) phi_{n-2}
    
    <phi_m, x^2 phi_n> = delta_{m,n+2} * 2^{n+2}(n+2)!/sqrt(2pi) + (2n+1) delta_{mn} * 2^n n!/sqrt(2pi) + n(n-1) delta_{m,n-2} * 2^{n-2}(n-2)!/sqrt(2pi)
    """
    # We need degrees 0, 4, 8, 12
    degrees = [0, 4, 8, 12]
    n_max = 14  # we need up to n+2 = 14
    
    # Compute factorials and powers of 2
    fact = np.zeros(n_max + 1)
    fact[0] = 1.0
    for i in range(1, n_max + 1):
        fact[i] = fact[i-1] * i
    
    pow2 = np.zeros(n_max + 1)
    pow2[0] = 1.0
    for i in range(1, n_max + 1):
        pow2[i] = pow2[i-1] * 2.0
    
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Norm: <phi_n, phi_n> = 2^n n! / sqrt(2pi)
    norm = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        norm[n] = pow2[n] * fact[n] / sqrt_2pi
    
    # Build matrices for the 4 basis functions
    M_0 = np.zeros((4, 4))
    M_pos = np.zeros((4, 4))  # <phi_m, x^2 phi_n>
    
    for i, mi in enumerate(degrees):
        for j, nj in enumerate(degrees):
            # M_0[i,j] = <phi_mi, phi_nj>
            if mi == nj:
                M_0[i, j] = norm[mi]
            else:
                M_0[i, j] = 0.0
            
            # M_pos[i,j] = <phi_mi, x^2 phi_nj>
            # x^2 phi_nj = phi_{nj+2} + (2*nj+1) phi_nj + nj*(nj-1) phi_{nj-2}
            val = 0.0
            # Term 1: phi_{nj+2}, contributes if mi == nj+2
            if mi == nj + 2 and nj + 2 <= n_max:
                val += norm[nj + 2]
            # Term 2: (2*nj+1) phi_nj, contributes if mi == nj
            if mi == nj:
                val += (2.0 * nj + 1.0) * norm[nj]
            # Term 3: nj*(nj-1) phi_{nj-2}, contributes if mi == nj-2 and nj >= 2
            if mi == nj - 2 and nj >= 2:
                val += nj * (nj - 1.0) * norm[nj - 2]
            
            M_pos[i, j] = val
    
    # For momentum: G_n = (-i)^n phi_n
    # <G_m, xi^2 G_n> = (-i)^m (-i)^n <phi_m, xi^2 phi_n>
    # (-i)^m (-i)^n = (-i)^{m+n} = (-1)^{(m+n)/2} * i^{(m+n) mod 2}
    # For even m, n: (-i)^{m+n} = (-1)^{(m+n)/2}
    # m, n in {0, 4, 8, 12}, all even
    # (m+n)/2: for (0,0):0, (0,4):2, (0,8):4, (0,12):6, (4,4):4, (4,8):6, (4,12):8, (8,8):8, (8,12):10, (12,12):12
    # All even, so (-1)^{(m+n)/2} = 1 for all pairs where (m+n)/2 is even
    # (m+n)/2 even means m+n divisible by 4
    # Let's just compute the phase factor
    M_mom = np.zeros((4, 4))
    for i, mi in enumerate(degrees):
        for j, nj in enumerate(degrees):
            # Phase: (-i)^mi * (-i)^nj = (-i)^(mi+nj)
            # For even mi, nj: (-i)^(mi+nj) = (-1)^((mi+nj)/2)
            phase = (-1.0) ** ((mi + nj) // 2)
            M_mom[i, j] = phase * M_pos[i, j]
    
    return M_0, M_pos, M_mom


def _bound_from_params(params, M_0, M_pos, M_mom):
    """
    Compute the uncertainty bound from parameters (a0, a4, a8).
    a12 is determined by g(0) = 0.
    
    g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    H0(0) = 1
    H4(0) = 3
    H8(0) = 105
    H12(0) = 945
    
    So: a0 + 3*a4 + 105*a8 + 945*a12 = 0
    a12 = -(a0 + 3*a4 + 105*a8) / 945
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    norm2 = a @ M_0 @ a
    if norm2 <= 1e-30 or not np.isfinite(norm2):
        return 1e18
    
    pos_var = a @ M_pos @ a
    if pos_var <= 0 or not np.isfinite(pos_var):
        return 1e18
    
    mom_var = a @ M_mom @ a
    if mom_var <= 0 or not np.isfinite(mom_var):
        return 1e18
    
    vf = pos_var / norm2
    vF = mom_var / norm2
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_and_grad(params, M_0, M_pos, M_mom):
    """
    Compute bound and analytic gradient.
    
    bound = sqrt(P * Q / N^2) where:
    N = a^T M_0 a
    P = a^T M_pos a  
    Q = a^T M_mom a
    
    a = [a0, a4, a8, a12] where a12 = -(a0 + 3*a4 + 105*a8)/945
    
    da/dparams (4x3):
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [-1/945, -3/945, -105/945]]
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N = a @ M_0 @ a
    P = a @ M_pos @ a
    Q = a @ M_mom @ a
    
    if N <= 1e-30 or P <= 0 or Q <= 0:
        if not np.isfinite(N) or not np.isfinite(P) or not np.isfinite(Q):
            return 1e18, np.array([1e18, 1e18, 1e18])
        return 1e18, np.array([0.0, 0.0, 0.0])
    
    bound = np.sqrt(P * Q / (N * N))
    
    # da/dparams (4x3)
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    # Compute M @ a for each matrix
    M0a = M_0 @ a
    Mposa = M_pos @ a
    Mmoma = M_mom @ a
    
    # dN/dparams = 2 * (da_dparams @ M0a)
    dN = 2.0 * (da_dparams @ M0a)
    dP = 2.0 * (da_dparams @ Mposa)
    dQ = 2.0 * (da_dparams @ Mmoma)
    
    # d(P*Q/N^2)/dparams = (dP*Q + P*dQ)/N^2 - 2*P*Q*dN/N^3
    dPQN2 = (dP * Q + P * dQ) / (N * N) - 2.0 * P * Q * dN / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(3)
    
    return bound, grad


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the uncertainty bound.
    The a12 coefficient is fixed by the constraint g(0) = 0.
    """
    try:
        M_0, M_pos, M_mom = _hermite_moments()
    except Exception:
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _obj(params):
        val, _ = _bound_and_grad(params, M_0, M_pos, M_mom)
        return val
    
    def _obj_grad(params):
        val, grad = _bound_and_grad(params, M_0, M_pos, M_mom)
        return val, grad
    
    # Generate diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.8, 0.06, 0.0006]),
        np.array([-4.0, 0.2, 0.003]),
        np.array([0.5, -0.03, 0.0003]),
    ]
    
    rng = np.random.default_rng(12345)
    for i in range(20):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_obj_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 2: SLSQP polish
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(_obj, best_params, method='SLSQP',
                          options={'maxiter': 10000, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-17, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Try a few more random starts near the best
    if time.time() < _DEADLINE - 5:
        rng2 = np.random.default_rng(99999)
        for i in range(10):
            if time.time() > _DEADLINE - 3:
                break
            perturbation = rng2.normal(0, 0.1, 3)
            start = best_params + perturbation
            try:
                res = minimize(_obj_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
