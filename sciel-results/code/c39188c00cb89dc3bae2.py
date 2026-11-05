import time
import math
import numpy as np
from scipy.optimize import minimize

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _gradient_reflectance(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute the analytical gradient of the mean reflectance with respect to thicknesses.
    """
    n_layers = len(materials)
    n_wls = len(wls)
    
    if n_layers == 0:
        return np.zeros(0)

    beta = np.zeros((n_layers, n_wls))
    dbeta_dd = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        beta[i, :] = 2.0 * math.pi * n * d / wls
        dbeta_dd[i, :] = 2.0 * math.pi * n / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Store layer matrices for later use in gradient calculation
    # M_i = [[cos, i sin/n], [i n sin, cos]]
    # dM_i/dd = [[-sin * dbeta, i cos/n * dbeta], [i n cos * dbeta, -sin * dbeta]]
    
    # We need to compute M_total and its derivatives.
    # Efficient way: compute prefix and suffix products.
    
    # Prefix products P_i = M_1 ... M_{i-1}
    # Suffix products S_i = M_{i+1} ... M_N
    
    # Let's just compute the total M first to get R and r.
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)
    
    # Store the matrices for each layer
    layer_A = np.zeros((n_layers, n_wls), dtype=complex)
    layer_B = np.zeros((n_layers, n_wls), dtype=complex)
    layer_C = np.zeros((n_layers, n_wls), dtype=complex)
    layer_D = np.zeros((n_layers, n_wls), dtype=complex)
    
    dlayer_A = np.zeros((n_layers, n_wls), dtype=complex)
    dlayer_B = np.zeros((n_layers, n_wls), dtype=complex)
    dlayer_C = np.zeros((n_layers, n_wls), dtype=complex)
    dlayer_D = np.zeros((n_layers, n_wls), dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        db = dbeta_dd[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c
        
        dA = -s * db
        dB = 1j * c / n * db
        dC = 1j * n * c * db
        dD = -s * db
        
        layer_A[i] = A
        layer_B[i] = B
        layer_C[i] = C
        layer_D[i] = D
        
        dlayer_A[i] = dA
        dlayer_B[i] = dB
        dlayer_C[i] = dC
        dlayer_D[i] = dD

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    # Gradient of R with respect to d_i
    # R = r * conj(r)
    # dR/dd_i = conj(r) * dr/dd_i + r * conj(dr/dd_i) = 2 * Re(conj(r) * dr/dd_i)
    
    # dr/dd_i = d/dY (r) * dY/dd_i
    # r = (n0 - Y) / (n0 + Y)
    # dr/dY = [-(n0+Y) - (n0-Y)] / (n0+Y)^2 = -2*n0 / (n0+Y)^2
    
    # dY/dd_i:
    # Y = (M21 + ns*M22) / (M11 + ns*M12)
    # Let Num = M21 + ns*M22, Den = M11 + ns*M12
    # dY = (dNum*Den - Num*dDen) / Den^2
    
    grad_R = np.zeros(n_layers)
    
    for i in range(n_layers):
        # Compute dM/dd_i
        # M = P_i * M_i * S_i
        # We need P_i and S_i.
        # To avoid O(N^2) complexity in Python loops for each gradient component,
        # we can precompute prefixes and suffixes.
        
        # Prefix P_i: product of matrices 0 to i-1
        # Suffix S_i: product of matrices i+1 to N-1
        
        # Let's compute prefixes and suffixes once for all i.
        # This is O(N * N_wls) operations, which is fine.
        
        # Actually, doing it inside the loop for each i is O(N^2).
        # Let's precompute.
        
        # Prefixes: P[i] = M_0 ... M_{i-1}
        # Suffixes: S[i] = M_{i+1} ... M_{N-1}
        
        # P[0] = I
        # P[i+1] = P[i] * M_i
        # S[N-1] = I
        # S[i-1] = S[i] * M_i
        
        P11 = np.ones((n_layers+1, n_wls), dtype=complex)
        P12 = np.zeros((n_layers+1, n_wls), dtype=complex)
        P21 = np.zeros((n_layers+1, n_wls), dtype=complex)
        P22 = np.ones((n_layers+1, n_wls), dtype=complex)
        
        for i in range(n_layers):
            P11[i+1] = P11[i] * layer_A[i] + P12[i] * layer_C[i]
            P12[i+1] = P11[i] * layer_B[i] + P12[i] * layer_D[i]
            P21[i+1] = P21[i] * layer_A[i] + P22[i] * layer_C[i]
            P22[i+1] = P21[i] * layer_B[i] + P22[i] * layer_D[i]
            
        S11 = np.ones((n_layers+1, n_wls), dtype=complex)
        S12 = np.zeros((n_layers+1, n_wls), dtype=complex)
        S21 = np.zeros((n_layers+1, n_wls), dtype=complex)
        S22 = np.ones((n_layers+1, n_wls), dtype=complex)
        
        for i in range(n_layers-1, -1, -1):
            S11[i] = S11[i+1] * layer_A[i] + S12[i+1] * layer_C[i]
            S12[i] = S11[i+1] * layer_B[i] + S12[i+1] * layer_D[i]
            S21[i] = S21[i+1] * layer_A[i] + S22[i+1] * layer_C[i]
            S22[i] = S21[i+1] * layer_B[i] + S22[i+1] * layer_D[i]
            
        # Now for layer i:
        # dM_i = [[dA, dB], [dC, dD]]
        # dM = P_i * dM_i * S_i
        
        # P_i is P[i], S_i is S[i+1]
        
        dM11 = (P11[i] * dlayer_A[i] + P12[i] * dlayer_C[i]) * S11[i+1] + \
               (P11[i] * dlayer_B[i] + P12[i] * dlayer_D[i]) * S21[i+1]
        dM12 = (P11[i] * dlayer_A[i] + P12[i] * dlayer_C[i]) * S12[i+1] + \
               (P11[i] * dlayer_B[i] + P12[i] * dlayer_D[i]) * S22[i+1]
        dM21 = (P21[i] * dlayer_A[i] + P22[i] * dlayer_C[i]) * S11[i+1] + \
               (P21[i] * dlayer_B[i] + P22[i] * dlayer_D[i]) * S21[i+1]
        dM22 = (P21[i] * dlayer_A[i] + P22[i] * dlayer_C[i]) * S12[i+1] + \
               (P21[i] * dlayer_B[i] + P22[i] * dlayer_D[i]) * S22[i+1]
        
        Num = M21 + n_substrate * M22
        Den = M11 + n_substrate * M12
        
        dNum = dM21 + n_substrate * dM22
        dDen = dM11 + n_substrate * dM12
        
        dY = (dNum * Den - Num * dDen) / (Den * Den)
        
        dr_dY = -2.0 * n_ambient / ((n_ambient + Y) * (n_ambient + Y))
        dr = dr_dY * dY
        
        dR = 2.0 * np.real(np.conj(r) * dr)
        
        grad_R[i] = np.mean(dR)
        
    return grad_R

def design(problem):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Generate candidate material sequences
    sequences = []
    for length in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if length > max_layers:
            break
        seq = ['L' if i % 2 == 0 else 'H' for i in range(length)]
        sequences.append(seq)
        
        if length >= 4:
            seq_h = ['H' if i % 2 == 0 else 'L' for i in range(length)]
            sequences.append(seq_h)

    # Optimization
    for mats in sequences:
        if time.time() > deadline - 5.0:
            break
            
        n_layers = len(mats)
        lambda_gm = math.sqrt(band_start * band_end)
        
        # Initial guesses
        x0_gm = [lambda_gm / (4.0 * n_mat[m]) for m in mats]
        x0_am = [((band_start + band_end) / 2.0) / (4.0 * n_mat[m]) for m in mats]
        
        # Randomized starts
        np.random.seed(hash(tuple(mats)) % 2**32)
        x0_rand1 = [d * np.random.uniform(0.7, 1.3) for d in x0_gm]
        x0_rand2 = [np.random.uniform(0, max_layer) for _ in range(n_layers)]
        
        initial_guesses = [x0_gm, x0_am, x0_rand1, x0_rand2]
        
        for x0 in initial_guesses:
            if time.time() > deadline - 2.0:
                break
                
            total = sum(x0)
            if total > max_total:
                factor = max_total / total
                x0 = [d * factor for d in x0]
            
            # Ensure bounds
            x0 = [min(max(d, 0.0), max_layer) for d in x0]
            
            # Define objective and gradient functions for SLSQP
            def obj(x):
                return _reflectance_array(mats, list(x), n_ambient, n_substrate, wls, n_mat).mean() * 100.0
            
            def grad(x):
                return _gradient_reflectance(mats, list(x), n_ambient, n_substrate, wls, n_mat) * 100.0

            # Constraints
            # Sum of thicknesses <= max_total
            cons = ({'type': 'ineq', 'fun': lambda x: max_total - np.sum(x)})
            # Bounds
            bounds = [(0.0, max_layer) for _ in range(n_layers)]
            
            # Run SLSQP
            try:
                res = minimize(obj, x0, method='SLSQP', jac=grad, bounds=bounds, constraints=cons, options={'maxiter': 100, 'ftol': 1e-12})
                if res.success:
                    x_opt = res.x
                    val = obj(x_opt)
                    if val < best_val:
                        best_val = val
                        best_mats = mats
                        best_ths = list(x_opt)
            except Exception:
                pass

    if best_mats is None:
        # Fallback to simple quarter wave
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _reflectance_array(best_mats, best_ths, n_ambient, n_substrate, wls, n_mat).mean() * 100.0
        
    # Final constraint enforcement
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
