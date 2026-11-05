import numpy as np
from scipy.optimize import minimize
import time

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 110.0
    
    # Precompute bit masks for energy calculation
    # Qubit i is bit i (LSB is qubit 0)
    # State index s. Bit i is (s >> i) & 1
    
    # Precompute diagonal for ZZ terms
    # ZZ_diag[k] is array of length 2^n
    # Value is 1 if (b_k + b_{k+1}) is even, -1 if odd
    # Wait, Z = diag(1, -1). Z_k Z_{k+1} has eigenvalue (-1)^{b_k} * (-1)^{b_{k+1}} = (-1)^{b_k+b_{k+1}}.
    
    dim = 2**n
    states = np.arange(dim)
    
    # Precompute ZZ diagonals for each bond
    zz_diags = []
    for k in range(n-1):
        bit_k = (states >> k) & 1
        bit_k1 = (states >> (k+1)) & 1
        parity = (bit_k + bit_k1) % 2
        # (-1)^parity
        zz_diag = 1 - 2 * parity
        zz_diags.append(zz_diag)
        
    # Precompute X masks
    # For each qubit k, we need to sum psi[u] * psi[u ^ 2^k] for u where bit k is 0
    # Or just compute <X> for all k
    
    def simulate(theta):
        # theta shape (layers, n)
        # Initial state |0...0>
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0
        
        for l in range(layers):
            # Apply Ry to all qubits
            # Qubit i is axis n-1-i
            for i in range(n):
                theta_i = theta[l, i]
                c = np.cos(theta_i / 2)
                s = np.sin(theta_i / 2)
                # Ry = [[c, -s], [s, c]]
                # Axis for qubit i is n-1-i
                axis = n - 1 - i
                # Reshape to move axis to end? Or use tensordot?
                # Let's move axis to end
                # psi shape: (2, 2, ..., 2)
                psi_reshaped = psi.reshape([2]*n)
                # Move axis to position n-1 (end)
                # Current position is axis.
                # np.moveaxis(psi_reshaped, axis, n-1)
                psi_moved = np.moveaxis(psi_reshaped, axis, n-1)
                # Now last axis is the qubit
                # Apply matrix
                # psi_moved shape: (2, ..., 2, 2)
                # new_psi_moved[..., 0] = c * psi_moved[..., 0] - s * psi_moved[..., 1]
                # new_psi_moved[..., 1] = s * psi_moved[..., 0] + c * psi_moved[..., 1]
                
                # Vectorized update
                # psi_moved is complex
                new_psi_moved = np.empty_like(psi_moved)
                new_psi_moved[..., 0] = c * psi_moved[..., 0] - s * psi_moved[..., 1]
                new_psi_moved[..., 1] = s * psi_moved[..., 0] + c * psi_moved[..., 1]
                
                # Move axis back
                psi_reshaped = np.moveaxis(new_psi_moved, n-1, axis)
                psi = psi_reshaped.reshape(dim)
                
            # Apply CZ to adjacent pairs
            # Pairs (0,1), (1,2), ...
            for i in range(n-1):
                # Qubit i and i+1
                axis_i = n - 1 - i
                axis_i1 = n - 1 - (i+1)
                
                # Move axis_i1 to 0, axis_i to 1?
                # They are adjacent.
                # Let's move them to the front.
                # np.moveaxis(psi, [axis_i1, axis_i], [0, 1])
                # Note: order matters.
                # If we move axis_i1 to 0 and axis_i to 1, the relative order is preserved if we are careful.
                # Actually, easier:
                # Reshape to group these two.
                
                psi_reshaped = psi.reshape([2]*n)
                # Move axis_i1 to 0, axis_i to 1
                # The other axes stay in relative order?
                # np.moveaxis moves axes to new positions.
                # Let's try:
                psi_moved = np.moveaxis(psi_reshaped, [axis_i1, axis_i], [0, 1])
                # Now psi_moved[0] is qubit i+1, psi_moved[1] is qubit i.
                # Reshape to (-1, 2, 2)
                psi_grouped = psi_moved.reshape(-1, 2, 2)
                
                # CZ matrix: diag(1, 1, 1, -1)
                # Basis: |00>, |01>, |10>, |11>
                # qubit i+1 is first index (row), qubit i is second index (col)?
                # Wait, standard tensor product order in reshape:
                # If we have axes A, B. Reshape to (N, 2, 2).
                # Element [x, a, b] corresponds to A=a, B=b.
                # Here A is qubit i+1, B is qubit i.
                # So index 0 is qubit i+1=0, index 1 is qubit i=0.
                # CZ applies -1 if both are 1.
                # So element [x, 1, 1] gets -1.
                
                psi_grouped[:, 1, 1] *= -1
                
                psi_moved = psi_grouped.reshape(psi_moved.shape)
                psi_reshaped = np.moveaxis(psi_moved, [0, 1], [axis_i1, axis_i])
                psi = psi_reshaped.reshape(dim)
                
        return psi

    def energy(theta):
        psi = simulate(theta)
        # Compute <H>
        # H = -J sum ZZ - h sum X
        
        e_ee = 0.0
        for k in range(n-1):
            # <psi| ZZ_k |psi> = sum |psi[s]|^2 * zz_diag[k][s]
            probs = np.abs(psi)**2
            e_ee += np.dot(probs, zz_diags[k])
            
        e_x = 0.0
        for k in range(n):
            mask = 1 << k
            # <psi| X_k |psi>
            # Sum over s where bit k is 0: psi[s] * conj(psi[s ^ mask]) + conj(psi[s]) * psi[s ^ mask]
            # = 2 Re( sum_{s: bit k=0} psi[s] * conj(psi[s ^ mask]) )
            # Identify indices where bit k is 0
            # (states >> k) & 1 == 0
            idx_0 = states[(states >> k) & 1 == 0]
            idx_1 = idx_0 + mask
            # psi[idx_0] * conj(psi[idx_1])
            terms = psi[idx_0] * np.conj(psi[idx_1])
            e_x += 2 * np.real(np.sum(terms))
            
        total_energy = -coupling * e_ee - field * e_x
        return total_energy

    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        return energy(theta)
        
    best_energy = np.inf
    best_theta = None
    
    num_restarts = 15
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            theta0 = np.zeros((layers, n))
            theta0[-1] = np.pi / 2
        else:
            theta0 = np.random.uniform(-np.pi, np.pi, (layers, n))
            
        try:
            res = minimize(objective, theta0.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
            
    if best_theta is None:
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
        
    return best_theta
