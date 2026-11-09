# EVOLVE-BLOCK-START
import numpy as np
import time

def _ry(theta):
    """Ry rotation matrix for a single qubit."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _cz():
    """CZ gate matrix."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=np.complex128)

def _apply_cz(state, n, q1, q2):
    """Apply CZ between qubits q1 and q2 (q1 < q2) on state of n qubits.
    Qubit i corresponds to bit i (little-endian: qubit 0 is least significant).
    """
    dim = 1 << n
    # We need to permute the state to bring q1 and q2 to the first two positions,
    # apply CZ, then permute back.
    # Actually, for small n we can just build the full CZ operator.
    # But that's expensive. Let's use a more efficient approach.
    
    # For efficiency with small n, let's just build the full operator.
    # For n=4, dim=16; for n=6, dim=64. This is fine.
    
    # Build CZ operator on qubits q1 and q2
    # The CZ operator acts as identity on all qubits except q1 and q2, where it's diag(1,1,1,-1)
    # We need to construct this in the full Hilbert space.
    
    # A simpler approach: iterate over basis states
    # For each basis state |b>, if bits q1 and q2 are both 1, flip sign
    new_state = state.copy()
    for i in range(dim):
        if ((i >> q1) & 1) and ((i >> q2) & 1):
            new_state[i] = -state[i]
    return new_state

def _apply_ry(state, n, qubit, theta):
    """Apply Ry(theta) to qubit on state of n qubits."""
    dim = 1 << n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    new_state = np.zeros_like(state)
    for i in range(dim):
        bit = (i >> qubit) & 1
        if bit == 0:
            # |0> -> c|0> + s|1>
            i0 = i
            i1 = i | (1 << qubit)
            new_state[i0] += c * state[i]
            new_state[i1] += s * state[i]
        else:
            # |1> -> -s|0> + c|1>
            i1 = i
            i0 = i & ~(1 << qubit)
            new_state[i0] += -s * state[i]
            new_state[i1] += c * state[i]
    return new_state

def _circuit_state(n, layers, params):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, n, q, params[layer, q])
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, n, q, q + 1)
    
    return state

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X_i terms
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            H[b, b_flip] += -field
            H[b_flip, b] += -field  # X is Hermitian, but we're adding to off-diagonals
            # Wait, X_i |b> = |b^flip>. So H = ... - field * X_i.
            # <b| X_i |b'> = delta_{b, b^flip}
            # So H[b, b'] += -field if b = b'^flip, i.e., b' = b^flip
            # Let me redo this.
    
    # Let me rebuild properly
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            H[b, b_flip] += -field
    
    return H

def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, coupling, field, layers):
    """Compute the objective: (circuit_energy - ground_energy) / n"""
    state = _circuit_state(n, layers, params)
    H = _hamiltonian(n, coupling, field)
    e_circuit = _energy(state, H)
    e_ground = _exact_ground_energy(n, coupling, field)
    return (e_circuit - e_ground) / n

def _nelder_mead(objective, x0, maxiter=200, tol=1e-8):
    """Simple Nelder-Mead implementation"""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.05 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    for it in range(maxiter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute ground energy
    e_ground = _exact_ground_energy(n, coupling, field)
    H = _hamiltonian(n, coupling, field)
    
    def obj(params):
        state = _circuit_state(n, layers, params)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    # Initial guess: all zeros (product |0> state)
    x0 = np.zeros((layers, n))
    
    # Multi-restart with different initial conditions
    best_params = x0.copy()
    best_val = obj(x0)
    
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = np.random.uniform(-np.pi/2, np.pi/2, (layers, n))
        
        # Flatten for Nelder-Mead
        x_flat = x_init.flatten()
        
        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))
        
        x_opt, f_opt = _nelder_mead(obj_flat, x_flat, maxiter=100, tol=1e-7)
        
        if f_opt < best_val:
            best_val = f_opt
            best_params = x_opt.reshape((layers, n))
        
        if time.time() > deadline:
            break
    
    # Final check
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    return best_params
# EVOLVE-BLOCK-END
