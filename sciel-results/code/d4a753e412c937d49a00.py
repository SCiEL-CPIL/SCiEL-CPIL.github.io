import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _precompute_super_terms(a, adag, n):
    I3 = np.eye(3, dtype=complex)
    anharmonicity = -2 * np.pi * 0.25
    H0 = anharmonicity * 0.5 * (n @ (n - I3))
    H0_super = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
    
    drive_I = 0.5 * (a + adag)
    drive_Q = 0.5j * (a - adag)
    drive_I_super = -1j * (np.kron(drive_I, I3) - np.kron(I3, drive_I.T))
    drive_Q_super = -1j * (np.kron(drive_Q, I3) - np.kron(I3, drive_Q.T))
    
    det_super = -1j * (np.kron(n, I3) - np.kron(I3, n.T))
    
    L1 = a / np.sqrt(25000.0)
    L2 = n * np.sqrt(2.0 / 40000.0)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    diss_super = L1_super + L2_super
    
    return H0_super, drive_I_super, drive_Q_super, det_super, diss_super

def _legendre_basis(t, n_poly=8):
    """Build Legendre polynomial basis evaluated at t (scaled to [-1, 1])."""
    B = np.zeros((len(t), n_poly))
    B[:, 0] = 1.0
    B[:, 1] = t
    for k in range(2, n_poly):
        B[:, k] = ((2 * k - 1) * t * B[:, k-1] - (k - 1) * B[:, k-2]) / k
    return B

def _pulse_from_legendre(coeff_I, coeff_Q, B):
    """Convert Legendre coefficients to I/Q amplitudes."""
    I = B @ coeff_I
    Q = B @ coeff_Q
    return np.column_stack([I, Q])

def _enforce_constraints(IQ):
    """Enforce amplitude and slew constraints."""
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4):
        scale = 0.4 / np.max(amps) * 0.999
        IQ = IQ * scale
    
    max_d = 0.12 * 0.5 * 0.999
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        diff = np.linalg.norm(IQ[i] - prev)
        if diff > max_d:
            if diff > 1e-15:
                IQ[i] = prev + (max_d / diff) * (IQ[i] - prev)
            else:
                IQ[i] = prev.copy()
    
    if np.linalg.norm(IQ[-1]) > max_d:
        if np.linalg.norm(IQ[-1]) > 1e-15:
            IQ[-1] *= max_d / np.linalg.norm(IQ[-1])
        else:
            IQ[-1] = np.array([0.0, 0.0])
    
    return IQ

def _valid(IQ):
    """Check if pulse satisfies all constraints."""
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        if np.linalg.norm(IQ[i] - prev) / 0.5 > 0.12 + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) / 0.5 > 0.12 + 1e-12:
        return False
    return True

def _simulate_pulse_fast(I, Q, super_terms):
    """Simulate pulse and compute infidelity for all 9 cases."""
    H0_super, drive_I_super, drive_Q_super, det_super, diss_super = super_terms
    dt = 0.5
    
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [1/np.sqrt(2), 1j/np.sqrt(2), 0],
        [1/np.sqrt(2), -1j/np.sqrt(2), 0]
    ], dtype=complex)
    
    det_values = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scale_values = [0.97, 1.0, 1.03]
    
    infs = np.zeros(9)
    
    for di, det in enumerate(det_values):
        for si, sc in enumerate(scale_values):
            idx = di * 3 + si
            for k in range(6):
                psi = states[k]
                rho = np.outer(psi, psi.conj()).flatten()
                for i in range(80):
                    L = H0_super + det * det_super + sc * (I[i] * drive_I_super + Q[i] * drive_Q_super) + diss_super
                    rho = expm(L * dt) @ rho
                rho = rho.reshape(3, 3)
                psi_t = X_gate @ psi
                rho_t = np.outer(psi_t, psi_t.conj())
                infs[idx] += (1 - np.real(np.trace(rho_t @ rho))) / 6.0
    
    return np.mean(infs) + np.max(infs)

def _score_fast(IQ, super_terms):
    """Score a pulse (80, 2) array."""
    I = IQ[:, 0]
    Q = IQ[:, 1]
    return _simulate_pulse_fast(I, Q, super_terms)

def _make_seed_pulse():
    """Create initial DRAG-style Gaussian pulse."""
    t = np.arange(80) * 0.5
    sigma = 40.0 / 6.0
    env = np.exp(-(t - 20) ** 2 / (2 * sigma ** 2))
    env = np.maximum(env - env[0] - env[-1], 0.0)
    area = np.sum(env) * 0.5
    if area > 1e-10:
        env *= np.pi / area
    I = env
    Q = -0.3 * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    return _enforce_constraints(IQ)

def _fit_legendre(IQ, B):
    """Fit Legendre coefficients to a given pulse."""
    I = IQ[:, 0]
    Q = IQ[:, 1]
    coeff_I, _, _, _ = np.linalg.lstsq(B, I, rcond=None)
    coeff_Q, _, _, _ = np.linalg.lstsq(B, Q, rcond=None)
    return np.concatenate([coeff_I, coeff_Q])

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    a, adag, n = _build_matrices()
    super_terms = _precompute_super_terms(a, adag, n)
    
    # Build Legendre basis on scaled time [0, 1]
    t = np.arange(80) * 0.5 / 40.0  # Scale to [0, 1]
    x_scaled = 2.0 * t - 1.0  # Scale to [-1, 1]
    B = _legendre_basis(x_scaled, n_poly=8)
    
    # Create seed pulse and fit to Legendre basis
    seed_IQ = _make_seed_pulse()
    seed_coeff = _fit_legendre(seed_IQ, B)
    
    # Compute seed score
    seed_score = _score_fast(seed_IQ, super_terms)
    
    best_coeff = seed_coeff.copy()
    best_score = seed_score
    best_IQ = seed_IQ.copy()
    
    # Phase 1: Optimize 16 Legendre coefficients (8 for I, 8 for Q)
    if time.time() < deadline - 60:
        def legendre_objective(c):
            if time.time() > deadline - 30:
                return 1e10
            coeff_I = c[:8]
            coeff_Q = c[8:16]
            IQ = _pulse_from_legendre(coeff_I, coeff_Q, B)
            IQ = _enforce_constraints(IQ)
            if not _valid(IQ):
                return 1e10
            return _score_fast(IQ, super_terms)
        
        try:
            res = minimize(legendre_objective, seed_coeff, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 300, 'disp': False})
            if res.fun < 1e9 and res.fun < best_score:
                best_score = res.fun
                best_coeff = res.x.copy()
                coeff_I = best_coeff[:8]
                coeff_Q = best_coeff[8:16]
                best_IQ = _pulse_from_legendre(coeff_I, coeff_Q, B)
                best_IQ = _enforce_constraints(best_IQ)
        except Exception:
            pass
    
    # Phase 2: Refine with Nelder-Mead on the same basis
    if time.time() < deadline - 40:
        def legendre_objective2(c):
            if time.time() > deadline - 20:
                return 1e10
            coeff_I = c[:8]
            coeff_Q = c[8:16]
            IQ = _pulse_from_legendre(coeff_I, coeff_Q, B)
            IQ = _enforce_constraints(IQ)
            if not _valid(IQ):
                return 1e10
            return _score_fast(IQ, super_terms)
        
        try:
            res2 = minimize(legendre_objective2, best_coeff, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < 1e9 and res2.fun < best_score:
                best_score = res2.fun
                best_coeff = res2.x.copy()
                coeff_I = best_coeff[:8]
                coeff_Q = best_coeff[8:16]
                best_IQ = _pulse_from_legendre(coeff_I, coeff_Q, B)
                best_IQ = _enforce_constraints(best_IQ)
        except Exception:
            pass
    
    # Phase 3: Try a different basis (Chebyshev) for diversity
    if time.time() < deadline - 20:
        n_basis = 8
        B_cheb = np.zeros((80, n_basis))
        B_cheb[:, 0] = 1.0
        B_cheb[:, 1] = x_scaled
        for k in range(2, n_basis):
            B_cheb[:, k] = 2 * x_scaled * B_cheb[:, k-1] - B_cheb[:, k-2]
        
        # Fit current best to Chebyshev basis
        coeff_cheb_I, _, _, _ = np.linalg.lstsq(B_cheb, best_IQ[:, 0], rcond=None)
        coeff_cheb_Q, _, _, _ = np.linalg.lstsq(B_cheb, best_IQ[:, 1], rcond=None)
        coeff0 = np.concatenate([coeff_cheb_I, coeff_cheb_Q])
        
        def cheb_objective(c):
            if time.time() > deadline - 5:
                return 1e10
            I_new = B_cheb @ c[:n_basis]
            Q_new = B_cheb @ c[n_basis:2*n_basis]
            IQ_new = np.column_stack([I_new, Q_new])
            IQ_new = _enforce_constraints(IQ_new)
            if not _valid(IQ_new):
                return 1e10
            return _score_fast(IQ_new, super_terms)
        
        try:
            res3 = minimize(cheb_objective, coeff0, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12})
            if res3.fun < 1e9 and res3.fun < best_score:
                best_score = res3.fun
                I_new = B_cheb @ res3.x[:n_basis]
                Q_new = B_cheb @ res3.x[n_basis:2*n_basis]
                best_IQ = np.column_stack([I_new, Q_new])
                best_IQ = _enforce_constraints(best_IQ)
        except Exception:
            pass
    
    # Final validation
    if not _valid(best_IQ):
        best_IQ = _enforce_constraints(best_IQ)
    
    return best_IQ
