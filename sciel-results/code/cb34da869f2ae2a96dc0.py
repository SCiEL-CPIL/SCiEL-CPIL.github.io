import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 255.0

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            op = _u3_matrix(th, ph, lm)
            full = np.array([[1]], dtype=complex)
            for k in range(qubits):
                full = np.kron(full, op if k == q else eye2)
            U = full @ U
        else:
            _, c, t = g
            n_bits = qubits
            idx = np.arange(n)
            cb = (idx >> (n_bits - 1 - c)) & 1
            perm = idx ^ (cb << (n_bits - 1 - t))
            U = U[perm, :]
    return U

def _dist(t, u):
    n = t.shape[0]
    return 1.0 - min(1.0, abs(np.trace(t.conj().T @ u)) / n)

def _count_cx(gates):
    return sum(1 for g in gates if g[0] == "cx")

def _detect_known(target, qubits):
    tol = 1e-8
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=tol):
            return "cnot"
        iswap = 0.5*np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if np.allclose(target, iswap, atol=tol):
            return "iswap"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=tol):
            return "swap"
    if qubits == 3:
        toff = np.zeros((8,8), dtype=complex)
        for i in range(8):
            b0, b1, b2 = (i>>2)&1, (i>>1)&1, i&1
            if b0 and b1:
                toff[i, i^1] = 1
            else:
                toff[i, i] = 1
        if np.allclose(target, toff, atol=tol):
            return "toffoli"
    return "generic"

def _exact_toffoli(qubits):
    gates = []
    gates.append(["u3", 2, math.pi/2, 0, 0])
    gates.append(["cx", 2, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi/2, 0, 0])
    return gates

def _iswap_exact():
    gates = []
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, math.pi/2])
    gates.append(["u3", 1, 0, 0, math.pi/2])
    return gates

def _optimal_u3(A):
    """Find (theta, phi, lam) such that u3(theta,phi,lam) best approximates A (2x2)."""
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    sqrt_det = np.sqrt(det_A)
    if abs(sqrt_det) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    c_val = A[0,0] / sqrt_det
    s_val = A[1,0] / sqrt_det
    
    c_abs = abs(c_val)
    s_abs = abs(s_val)
    norm = math.sqrt(c_abs**2 + s_abs**2)
    if norm < 1e-15:
        return (0.0, 0.0, 0.0)
    
    theta_half = math.atan2(s_abs, c_abs)
    theta = 2.0 * theta_half
    
    if s_abs > 1e-15:
        e_iphi = s_val / s_abs
        phi = np.angle(e_iphi)
    else:
        phi = 0.0
    
    if s_abs > 1e-15:
        e_ilam = -(A[0,1] / sqrt_det) / s_abs
        lam = np.angle(e_ilam)
    else:
        lam = 0.0
    
    return (float(theta), float(phi), float(lam))

def _apply_u3_to_layer(U, qubit, theta, phi, lam, qubits):
    n = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    op = _u3_matrix(theta, phi, lam)
    full = np.array([[1]], dtype=complex)
    for k in range(qubits):
        full = np.kron(full, op if k == qubit else eye2)
    return full @ U

def _apply_cx(U, control, target, qubits):
    n = 2 ** qubits
    idx = np.arange(n)
    cb = (idx >> (qubits - 1 - control)) & 1
    perm = idx ^ (cb << (qubits - 1 - target))
    return U[perm, :]

def _build_from_params(params, cnot_template, qubits):
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    U = np.eye(n, dtype=complex)
    param_idx = 0
    eye2 = np.eye(2, dtype=complex)
    for layer in range(num_layers):
        for q in range(qubits):
            th = params[param_idx]
            ph = params[param_idx+1]
            lm = params[param_idx+2]
            R = _u3_matrix(th, ph, lm)
            full = np.array([[1]], dtype=complex)
            for k in range(qubits):
                full = np.kron(full, R if k == q else eye2)
            U = full @ U
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            idx = np.arange(n)
            cb = (idx >> (qubits - 1 - c)) & 1
            perm = idx ^ (cb << (qubits - 1 - t))
            U = U[perm, :]
    return U

def _cost_fn(params, target, cnot_template, qubits):
    n = 2 ** qubits
    U = _build_from_params(params, cnot_template, qubits)
    tr_val = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(tr_val) / n)

def _optimize_circuit(target, qubits, cnot_template, max_sweeps=50):
    """Coordinate descent with analytic u3 optimization per coordinate."""
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * qubits
    num_params = num_u3 * 3
    params = np.zeros(num_params, dtype=float)
    
    # Precompute CNOT permutation matrices
    cx_perms = []
    for c, t in cnot_template:
        idx = np.arange(n)
        cb = (idx >> (qubits - 1 - c)) & 1
        perm = idx ^ (cb << (qubits - 1 - t))
        cx_perms.append(perm)
    
    best_cost = _cost_fn(params, target, cnot_template, qubits)
    best_params = params.copy()
    
    for sweep in range(max_sweeps):
        if _remaining() < 5.0:
            break
        
        improved = False
        for layer in range(num_layers):
            for q in range(qubits):
                base_idx = layer * qubits * 3 + q * 3
                th0 = params[base_idx]
                ph0 = params[base_idx+1]
                lm0 = params[base_idx+2]
                
                # Build circuit with current params, but replace this u3 with identity
                # to get the "context" matrix
                # Context = (all gates after this u3) @ target @ (all gates before this u3)^dagger
                # Then we want u3 to maximize |tr(u3 @ context)|
                
                # Build the "before" part: all u3s and CNOTs before position (layer, q)
                before_params = params.copy()
                before_params[base_idx:base_idx+3] = [0.0, 0.0, 0.0]
                
                # We need to compute: U_before^dagger @ target @ U_after
                # where U_before includes all gates up to (but not including) this u3
                # and U_after includes all gates from (but not including) this u3 onward
                
                # This is expensive to compute naively. Let's use a different approach:
                # Compute the full circuit with this u3 as identity, call it U_id
                # Then U_id = U_before @ I_q @ U_after
                # We want to find u3 such that U_before @ u3 @ U_after ≈ target
                # i.e., u3 ≈ U_before^dagger @ target @ U_after^dagger
                
                # Let's compute U_before and U_after separately
                # U_before: all layers before `layer`, plus u3s in layer before q
                # U_after: CNOT in layer (if any), plus u3s in layer after q, plus all later layers
                
                # For efficiency, let's just do a grid search over theta, phi, lam
                # for this coordinate. Since the space is 3D, we can do a coarse-to-fine search.
                
                # Actually, let's use the analytic approach: 
                # The optimal u3 for a given 2x2 target B is given by _optimal_u3(B)
                # We need to extract the effective 2x2 target for this u3.
                
                # Method: compute U_id (circuit with this u3 as identity)
                U_id = _build_from_params(before_params, cnot_template, qubits)
                
                # Now U_id = U_before @ I @ U_after
                # We want U_before @ u3 @ U_after = target
                # => u3 = U_before^dagger @ target @ U_after^dagger
                # But we don't have U_before and U_after separately.
                
                # Alternative: U_id^dagger @ target = (U_before @ I @ U_after)^dagger @ target
                # = U_after^dagger @ U_before^dagger @ target
                # This isn't directly useful.
                
                # Let's just do a numerical search over the 3 parameters for this coordinate.
                # Coarse grid first, then refine.
                
                best_local_cost = best_cost
                best_local_params = [th0, ph0, lm0]
                
                # Coarse search: theta in [0, pi], phi in [0, 2pi], lam in [0, 2pi]
                # But this is 3D, so let's do it smarter.
                
                # Fix phi and lam, optimize theta analytically? No, it's coupled.
                
                # Let's do a random restart approach: try several random initializations
                # and do a few gradient steps.
                
                # Actually, let's just use the 2x2 block approach which is fast:
                # The effective 2x2 matrix for qubit q in the context of U_id is:
                # B = U_id^dagger @ target, then extract the 2x2 block for qubit q
                
                B_full = U_id.conj().T @ target
                
                # Extract 2x2 block for qubit q
                idx0 = 0
                idx1 = 1 << (qubits - 1 - q)
                
                B = np.array([[B_full[idx0, idx0], B_full[idx0, idx1]],
                              [B_full[idx1, idx0], B_full[idx1, idx1]]], dtype=complex)
                
                th, ph, lm = _optimal_u3(B)
                
                # Update and check cost
                params[base_idx] = th
                params[base_idx+1] = ph
                params[base_idx+2] = lm
                
                cost = _cost_fn(params, target, cnot_template, qubits)
                if cost < best_local_cost - 1e-12:
                    best_local_cost = cost
                    improved = True
                    if cost < best_cost:
                        best_cost = cost
                        best_params = params.copy()
                else:
                    # Revert
                    params[base_idx] = th0
                    params[base_idx+1] = ph0
                    params[base_idx+2] = lm0
        
        if not improved:
            break
        if best_cost < 1e-10:
            break
    
    return best_params, best_cost

def _params_to_gates(params, cnot_template, qubits):
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    gates = []
    param_idx = 0
    for layer in range(num_layers):
        for q in range(qubits):
            th = params[param_idx]
            ph = params[param_idx+1]
            lm = params[param_idx+2]
            if abs(th) < 1e-9 and abs(ph) < 1e-9 and abs(lm) < 1e-9:
                param_idx += 3
                continue
            gates.append(["u3", q, float(th), float(ph), float(lm)])
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    return gates

def _cleanup_gates(gates, target, qubits, budget):
    cleaned = []
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            if abs(th) > 1e-9 or abs(ph) > 1e-9 or abs(lm) > 1e-9:
                cleaned.append(g)
        else:
            cleaned.append(g)
    
    while _count_cx(cleaned) > budget and len(cleaned) > 0:
        for i in range(len(cleaned) - 1, -1, -1):
            if cleaned[i][0] == "cx":
                cleaned.pop(i)
                break
        else:
            break
    
    return cleaned

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    gate_type = _detect_known(target, qubits)

    if gate_type == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_type == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_type == "iswap" and qubits == 2:
        return _iswap_exact()
    elif gate_type == "toffoli" and qubits == 3:
        return _exact_toffoli(qubits)

    if qubits == 2:
        num_cx = min(budget, 3)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,0), (0,1), (1,0), (0,1), (1,0)]
        cnot_template = pairs[:num_cx]
    else:
        num_cx = min(budget, 6)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        cnot_template = pairs[:num_cx]

    max_sweeps = 30 if qubits == 2 else 20
    params, cost = _optimize_circuit(target, qubits, cnot_template, max_sweeps=max_sweeps)
    gates = _params_to_gates(params, cnot_template, qubits)
    gates = _cleanup_gates(gates, target, qubits, budget)

    if cost > 1e-6 and _remaining() > 30.0:
        if qubits == 2:
            alt_pairs = [(1,0), (0,1), (1,0), (0,1), (1,0), (0,1)]
        else:
            alt_pairs = [(1,2), (0,1), (1,2), (0,1), (1,2), (0,1)]
        alt_template = alt_pairs[:num_cx]
        params2, cost2 = _optimize_circuit(target, qubits, alt_template, max_sweeps=max_sweeps)
        if cost2 < cost:
            gates2 = _params_to_gates(params2, alt_template, qubits)
            gates2 = _cleanup_gates(gates2, target, qubits, budget)
            gates = gates2

    return gates
