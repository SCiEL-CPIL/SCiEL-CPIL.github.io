import numpy as np
import scipy.optimize as opt
import time

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # 2. Define objective function
    def objective(theta_flat):
        return _circuit_energy(theta_flat, n, layers, H)
        
    # 3. Optimization
    best_energy = np.inf
    best_params = None
    
    # Initial guesses
    guesses = []
    guesses.append(np.full(layers * n, np.pi / 2))
    
    np.random.seed(0)
    for _ in range(10):
        guesses.append(np.random.uniform(0, 2 * np.pi, size=layers * n))
        
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(objective, guess, method='L-BFGS-B', 
                               bounds=[(0, 2*np.pi)] * (layers*n),
                               options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    return best_params.reshape((layers, n))

def _build_hamiltonian(n, coupling, field):
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    # Qubit i is bit i (LSB is 0).
    # We build kron from n-1 down to 0 so that Qubit 0 is the rightmost factor.
    for i in range(n-1):
        term = np.eye(1)
        for j in range(n-1, -1, -1):
            if j == i or j == i+1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H -= coupling * term
        
    # X terms: -field * sum X_i
    for i in range(n):
        term = np.eye(1)
        for j in range(n-1, -1, -1):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H -= field * term
        
    return H

def _apply_ry(psi, theta, qubit, n):
    axis = n - 1 - qubit
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    Ry = np.array([[c, -s], [s, c]], dtype=complex)
    
    psi_t = psi.reshape([2] * n)
    psi_t = np.moveaxis(psi_t, axis, 0)
    shape = psi_t.shape
    psi_2d = psi_t.reshape(2, -1)
    
    new_psi_2d = np.tensordot(Ry, psi_2d, axes=([1], [0]))
    new_psi_t = new_psi_2d.reshape(shape)
    new_psi_t = np.moveaxis(new_psi_t, 0, axis)
    
    return new_psi_t.reshape(-1)

def _apply_cz(psi, qubit1, qubit2, n):
    psi_flat = psi.copy()
    # qubit i is bit i.
    for state in range(2**n):
        b1 = (state >> qubit1) & 1
        b2 = (state >> qubit2) & 1
        if b1 == 1 and b2 == 1:
            psi_flat[state] *= -1
    return psi_flat

def _circuit_energy(theta_flat, n, layers, H):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    
    theta_reshaped = theta_flat.reshape((layers, n))
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            psi = _apply_ry(psi, theta_reshaped[l, i], i, n)
            
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                psi = _apply_cz(psi, i, i + 1, n)
                
    # Calculate energy
    # <psi| H |psi>
    H_psi = H @ psi
    energy = np.vdot(psi, H_psi).real
    return energy
