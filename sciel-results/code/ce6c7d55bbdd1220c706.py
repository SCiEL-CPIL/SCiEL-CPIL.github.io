import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Hermite polynomial coefficients (physicists' convention), stored as [x^0, x^2, x^4, ...]
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Precompute integrals
# I(n) = ∫ x^(2n) exp(-x²) dx = (2n-1)!! * sqrt(pi) / 2^n
# J(n) = ∫ x^(2n) exp(-2x²) dx = (2n-1)!! * sqrt(pi) / 2^(2n+0.5)

def _precompute_integrals():
    I = np.zeros(13)
    for n in range(13):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            I[n] = result * np.sqrt(np.pi) / (2.0 ** n)
    
    J = np.zeros(25)
    for n in range(25):
        if n == 0:
            J[n] = np.sqrt(np.pi / 2.0)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            J[n] = result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))
    
    return I, J

I_int, J_int = _precompute_integrals()

def _compute_P_coeffs(c0, c1, c2, c3):
    P = np.zeros(7)
    P[0] += c0 * H0_coeffs[0]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    return P

def _poly_square_moment(P):
    n = len(P)
    M = np.array([[I_int[i+j] for j in range(n)] for i in range(n)])
    return float(P @ M @ P)

def _poly_fourth_moment(P):
    n = len(P)
    Q = np.zeros(13)
    for i in range(n):
        for j in range(n):
            Q[i+j] += P[i] * P[j]
    result = 0.0
    for m in range(13):
        for nn in range(13):
            result += Q[m] * Q[nn] * J_int[m+nn]
    return result

def _objective_and_grad(c0, c1, c2):
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    L2_sq = _poly_square_moment(P)
    L4_4th = _poly_fourth_moment(P)
    
    if L2_sq <= 0 or L4_4th <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    dP_dc0 = np.zeros(7)
    dP_dc0[0] += 1.0
    for i in range(7):
        dP_dc0[i] += (-1/1039500.0) * H12_coeffs[i]
    
    dP_dc1 = np.zeros(7)
    for i in range(3):
        dP_dc1[i] += H4_coeffs[i]
    for i in range(7):
        dP_dc1[i] += (-12/1039500.0) * H12_coeffs[i]
    
    dP_dc2 = np.zeros(7)
    for i in range(5):
        dP_dc2[i] += H8_coeffs[i]
    for i in range(7):
        dP_dc2[i] += (-3360/1039500.0) * H12_coeffs[i]
    
    n = len(P)
    M = np.array([[I_int[i+j] for j in range(n)] for i in range(n)])
    
    dL2_sq_dc0 = 2.0 * float(P @ M @ dP_dc0)
    dL2_sq_dc1 = 2.0 * float(P @ M @ dP_dc1)
    dL2_sq_dc2 = 2.0 * float(P @ M @ dP_dc2)
    
    def _compute_dQ(dP, P):
        dQ = np.zeros(13)
        for i in range(7):
            for j in range(7):
                dQ[i+j] += 2.0 * dP[i] * P[j]
        return dQ
    
    dQ_dc0 = _compute_dQ(dP_dc0, P)
    dQ_dc1 = _compute_dQ(dP_dc1, P)
    dQ_dc2 = _compute_dQ(dP_dc2, P)
    
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    def _dL4_4th(dQ, Q):
        result = 0.0
        for m in range(13):
            for nn in range(13):
                result += 4.0 * dQ[m] * Q[nn] * J_int[m+nn]
        return result
    
    dL4_4th_dc0 = _dL4_4th(dQ_dc0, Q)
    dL4_4th_dc1 = _dL4_4th(dQ_dc1, Q)
    dL4_4th_dc2 = _dL4_4th(dQ_dc2, Q)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_dc0 = (0.5 * dL4_4th_dc0 / sqrt_L4 - dL2_sq_dc0 / L2_sq) * C4
    dC4_dc1 = (0.5 * dL4_4th_dc1 / sqrt_L4 - dL2_sq_dc1 / L2_sq) * C4
    dC4_dc2 = (0.5 * dL4_4th_dc2 / sqrt_L4 - dL2_sq_dc2 / L2_sq) * C4
    
    return C4, np.array([dC4_dc0, dC4_dc1, dC4_dc2])

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s limit
    
    # The objective is scale-invariant: C4(λc0, λc1, λc2) = C4(c0, c1, c2)
    # We can fix c0 = 1 and optimize over c1, c2
    # This reduces the search space from 3D to 2D
    
    def _obj_2d(params):
        c1, c2 = params
        c0 = 1.0
        val, _ = _objective_and_grad(c0, c1, c2)
        return val
    
    def _grad_2d(params):
        c1, c2 = params
        c0 = 1.0
        _, grad = _objective_and_grad(c0, c1, c2)
        return np.array([grad[1], grad[2]])
    
    # Known good starting point from current program
    best_c_full = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_C4, _ = _objective_and_grad(best_c_full[0], best_c_full[1], best_c_full[2])
    
    # Normalize to c0=1
    if best_c_full[0] != 0:
        best_2d = np.array([best_c_full[1]/best_c_full[0], best_c_full[2]/best_c_full[0]])
    else:
        best_2d = np.array([0.0, 0.0])
    
    # Phase 1: Global search with differential evolution on 2D space
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-10.0, 10.0), (-10.0, 10.0)]
            result_de = differential_evolution(
                _obj_2d,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=False
            )
            if result_de.fun < best_C4:
                best_C4 = result_de.fun
                best_2d = result_de.x
        except Exception:
            pass
    
    # Phase 2: Multi-start BFGS from multiple points
    rng = np.random.RandomState(42)
    starts = [best_2d.copy()]
    
    # Grid around known good point
    for dc1 in np.linspace(-5.0, 5.0, 11):
        for dc2 in np.linspace(-5.0, 5.0, 11):
            if time.time() > deadline - 30.0:
                break
            s = best_2d + np.array([dc1, dc2])
            starts.append(s)
    
    # Random starts
    for _ in range(30):
        s = best_2d + rng.normal(0, 3.0, 2)
        starts.append(s)
    
    # Evaluate and sort
    start_objs = []
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            obj = _obj_2d(s)
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    # Run BFGS from best starting points
    for obj, s in start_objs:
        if time.time() > deadline - 5.0:
            break
        if obj >= best_C4:
            continue
        try:
            result = minimize(
                _obj_2d,
                s,
                jac=_grad_2d,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-14}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_2d = result.x
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                _obj_2d,
                best_2d,
                jac=_grad_2d,
                method='BFGS',
                options={'maxiter': 1000, 'gtol': 1e-16}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_2d = result.x
        except Exception:
            pass
    
    # Convert back to 3 coefficients with c0 = 1
    c0 = 1.0
    c1 = best_2d[0]
    c2 = best_2d[1]
    
    return np.array([c0, c1, c2])
