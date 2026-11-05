"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize


def _build_hamiltonian(n, coupling, field):
    """Build sparse Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    dim = 1 << n
    diag_vals = np.zeros(dim)
    for state in range(dim):
        zz_sum = 0.0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_sum += z_i * z_ip1
        diag_vals[state] = -coupling * zz_sum
    
    rows, cols, data = [], [], []
    for i in range(dim):
        rows.append(i); cols.append(i); data.append(diag_vals[i])
    
    for i in range(n):
        mask = 1 << i
        for state in range(dim):
            other = state ^ mask
            if other > state:
                rows.append(state); cols.append(other); data.append(-field)
                rows.append(other); cols.append(state); data.append(-field)
    
    return csr_matrix((data, (rows, cols)), shape=(dim, dim))


def _circuit_state(params, n, layers):
    """Compute |psi> = U(params)|0>^n where U has Ry layers with CZ between."""
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry(theta_i) to qubit i for all i
        for i in range(n):
            theta = params[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # Ry = [[c, -s], [s, c]]
            # New amplitudes: a0' = c*a0 + s*a1, a1' = s*a0 - c*a1
            new_state = np.zeros_like(state)
            for basis in range(1 << n):
                bit = (basis >> i) & 1
                if bit == 0:
                    partner = basis | (1 << i)
                    new_state[basis] += c * state[basis] + s * state[partner]
                    new_state[partner] += s * state[basis] - c * state[partner]
                # else: already handled when bit=0 partner was processed
            # Actually this double-counts. Let me redo properly.
            new_state = np.zeros_like(state)
            for basis in range(1 << n):
                bit = (basis >> i) & 1
                if bit == 0:
                    partner = basis | (1 << i)
                    a0 = state[basis]
                    a1 = state[partner]
                    new_state[basis] = c * a0 + s * a1
                    new_state[partner] = s * a0 - c * a1
                else:
                    partner = basis & ~(1 << i)
                    # Already computed when we hit partner
                    pass
            # Need to handle both cases properly
            new_state = np.zeros_like(state)
            for basis in range(1 << n):
                bit = (basis >> i) & 1
                if bit == 0:
                    partner = basis | (1 << i)
                    a0 = state[basis]
                    a1 = state[partner]
                    new_state[basis] = c * a0 + s * a1
                    new_state[partner] = s * a0 - c * a1
            state = new_state
        
        # Apply CZ between adjacent pairs (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ: only |11> on qubits i,i+1 gets -1
                for basis in range(1 << n):
                    bit_i = (basis >> i) & 1
                    bit_ip1 = (basis >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[basis] *= -1
    
    return state


def _energy(params, n, coupling, field, layers, H):
    psi = _circuit_state(params, n, layers)
    Hpsi = H @ psi
    return np.vdot(psi, Hpsi).real


def angles(n, coupling, field, layers):
    H = _build_hamiltonian(n, coupling, field)
    
    # Exact ground energy via dense diagonalization (n<=6, dim<=64)
    H_dense = H.toarray()
    eigvals = np.linalg.eigvalsh(H_dense)
    exact_ground_energy = eigvals[0]
    
    # Initial guess: start near |+>^n
    x0 = np.zeros((layers, n))
    x0[-1] = np.pi / 2
    # Small perturbation to break symmetry
    np.random.seed(42)
    x0 += 0.2 * np.random.randn(layers, n)
    
    def objective(x):
        e = _energy(x, n, coupling, field, layers, H)
        return (e - exact_ground_energy) / n
    
    # Use L-BFGS-B with finite-difference gradient
    eps = 1e-5
    def grad(x):
        g = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            xp = x.copy(); xp[idx] += eps
            xm = x.copy(); xm[idx] -= eps
            g[idx] = (_energy(xp, n, coupling, field, layers, H) - 
                     _energy(xm, n, coupling, field, layers, H)) / (2 * eps)
        return g / n
    
    best_params = x0
    best_val = objective(x0)
    
    # Multiple restarts for robustness
    for restart in range(3):
        if restart == 0:
            start = x0
        else:
            start = x0 + 0.5 * np.random.randn(layers, n)
        
        result = minimize(objective, start, jac=grad, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7})
        
        val = objective(result.x)
        if val < best_val:
            best_val = val
            best_params = result.x
    
    if not np.all(np.isfinite(best_params)):
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
    
    return best_params
# EVOLVE-BLOCK-END
