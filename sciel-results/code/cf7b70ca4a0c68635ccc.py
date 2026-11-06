"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, eigh

_DEADLINE = time.time() + 115.0


def _time_left():
    return _DEADLINE - time.time()


class QAOASimulator:
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.dim = 1 << n
        self._build_hamiltonians()
        self._init_state = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)
        
    def _build_hamiltonians(self):
        # Build Cost Hamiltonian H_C = sum_{(u,v) in E} 0.5 * (1 - Z_u Z_v)
        # Identity part is 0.5 * |E| * I, which is a constant shift. 
        # For expectation value of cut, we want <sum 0.5(1-ZuZv)>.
        # Let's build the full matrix H_C.
        H_C = np.zeros((self.dim, self.dim), dtype=complex)
        
        # Z operator
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        # Precompute Pauli Z for each qubit as sparse-like or just build the full matrix
        # Since n is small (<=10), full matrix is fine.
        # Z_i = I ... I Z I ... I
        Z_ops = []
        for i in range(self.n):
            op = np.eye(1, dtype=complex)
            for j in range(self.n):
                if j == i:
                    op = np.kron(op, Z)
                else:
                    op = np.kron(op, I2)
            Z_ops.append(op)
            
        for u, v in self.edges:
            # Term: 0.5 * (I - Z_u Z_v)
            # Z_u Z_v is the tensor product of Z_u and Z_v
            # We can compute Z_u Z_v by multiplying the full matrices? 
            # Or better, construct it.
            # Actually, it's easier to just add the terms to H_C.
            # H_C = sum 0.5 I - 0.5 sum Z_u Z_v
            # The sum of 0.5 I is 0.5 * len(edges) * I_dim.
            # Let's accumulate the -0.5 Z_u Z_v part.
            
            # Construct Z_u Z_v
            # Z_u Z_v = (Z_u) (Z_v)
            # This is a diagonal matrix with entries +/- 1.
            # We can generate the diagonal directly for speed.
            pass
            
        # Faster construction of H_C diagonal part (since ZZ is diagonal)
        # H_C is diagonal in the computational basis? 
        # Z_u Z_v is diagonal. So H_C is diagonal.
        # <z| H_C |z> = sum_{(u,v)} 0.5 * (1 - z_u z_v) where z_i is +/- 1.
        # Wait, Z|0> = |0>, Z|1> = -|1>. So eigenvalue is 1 for 0, -1 for 1.
        # Let's map bit string to eigenvalue.
        # If bit is 0, Z eigenvalue is 1. If bit is 1, Z eigenvalue is -1.
        # z_u z_v is 1 if bits are same, -1 if different.
        # 1 - z_u z_v is 0 if same, 2 if different.
        # 0.5 * (1 - z_u z_v) is 0 if same, 1 if different.
        # This is exactly the cut indicator for that edge.
        
        diag = np.zeros(self.dim)
        for state_idx in range(self.dim):
            bits = np.array([(state_idx >> i) & 1 for i in range(self.n)])
            cut_val = 0.0
            for u, v in self.edges:
                if bits[u] != bits[v]:
                    cut_val += 1.0
            diag[state_idx] = cut_val
            
        self.H_C_diag = diag
        self.H_C = np.diag(diag).astype(complex)
        
        # Mixer Hamiltonian H_B = sum X_i
        # We don't need the full matrix, we can apply the unitary directly.
        # U_B(beta) = exp(-i beta sum X_i) = tensor product of exp(-i beta X)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # We will store the 2x2 matrix for the mixer.
        self.beta_mat = np.array([
            [np.cos(0), -1j*np.sin(0)], # Placeholder, will be updated in apply
            [-1j*np.sin(0), np.cos(0)]
        ], dtype=complex)
        
    def apply_mixer(self, state, beta):
        # Apply exp(-i beta sum X_i) to state.
        # R_x(beta) = cos(beta) I - i sin(beta) X = [[cos, -i sin], [-i sin, cos]]
        # For the specific case where the same rotation is applied to all qubits,
        # we can use a faster method based on the structure of the state.
        # 
        # Key insight: exp(-i beta sum X_i) |b> where b is a bitstring
        # = prod_i (cos(beta)|b_i> - i sin(beta)|1-b_i>)
        # 
        # This means each basis state |b> maps to a superposition over all 2^n states
        # with amplitudes that factorize. We can compute this efficiently.
        #
        # Alternative faster approach: use the fact that for uniform rotation on all qubits,
        # we can compute the result using the number of 1-bits in each basis state.
        #
        # Actually, the simplest fast method: precompute the full mixer matrix for this beta
        # and do a single matrix-vector multiply. For n=10, that's 1024x1024 @ 1024 = 1M flops,
        # which is fast in numpy.
        
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Build the full mixer matrix: Rx ⊗ Rx ⊗ ... ⊗ Rx
        # Rx = [[c, -1j*s], [-1j*s, c]]
        Rx = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
        
        # Build full matrix by iterative kron
        U = Rx
        for _ in range(self.n - 1):
            U = np.kron(U, Rx)
        
        return U @ state

    def apply_cost(self, state, gamma):
        # H_C is diagonal. 
        # exp(-i gamma H_C) is diagonal with entries exp(-i gamma H_C_diag[i])
        # state_new[i] = exp(-i gamma H_C_diag[i]) * state_old[i]
        phases = np.exp(-1j * gamma * self.H_C_diag)
        return state * phases

    def expected_cut(self, params):
        # params: [g1, b1, g2, b2]
        # Initial state |+>^n
        state = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)
        
        g1, b1, g2, b2 = params
        
        # Layer 1
        state = self.apply_cost(state, g1)
        state = self.apply_mixer(state, b1)
        
        # Layer 2
        state = self.apply_cost(state, g2)
        state = self.apply_mixer(state, b2)
        
        # Expected cut
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self.H_C_diag))
    
    def expected_cut_fast(self, g1, b1, g2, b2):
        """Faster version that avoids function call overhead."""
        state = self._init_state
        state = self.apply_cost(state, g1)
        state = self.apply_mixer(state, b1)
        state = self.apply_cost(state, g2)
        state = self.apply_mixer(state, b2)
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self.H_C_diag))

def angles(n, edges, p):
    p = int(p)
    if _time_left() < 2.0:
        return np.tile([0.35, 0.25], (p, 1))

    sim = QAOASimulator(n, edges)
    
    # Coarse grid search for initialization - use finer grid since evaluation is faster
    grid = np.linspace(0.0, np.pi, 12)
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if _time_left() < 8.0:
                        break
                    val = sim.expected_cut_fast(g1, b1, g2, b2)
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if _time_left() < 8.0:
                    break
            if _time_left() < 8.0:
                break
        if _time_left() < 8.0:
            break

    # Local refinement with multiple restarts
    if _time_left() > 15.0:
        def neg_cut(params):
            return -sim.expected_cut_fast(params[0], params[1], params[2], params[3])
        
        rng = np.random.default_rng(42)
        n_restarts = min(5, max(1, int(_time_left() / 20.0)))
        
        for restart in range(n_restarts):
            if _time_left() < 10.0:
                break
            
            if restart == 0:
                x0 = best_params.copy()
            else:
                # Perturb from best found so far
                x0 = best_params + rng.normal(0, 0.3, size=4)
                x0 = np.clip(x0, 0.0, np.pi)
            
            try:
                res = minimize(neg_cut, x0, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-7})
                if res.fun < neg_cut(best_params):
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if _time_left() > 5.0:
        def neg_cut(params):
            return -sim.expected_cut_fast(params[0], params[1], params[2], params[3])
        
        try:
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-9})
            if res.fun < neg_cut(best_params):
                best_params = res.x.copy()
        except Exception:
            pass

    best_params = np.asarray(best_params, dtype=float)
    best_params = np.clip(best_params, 0.0, np.pi)

    if p == 2:
        return best_params.reshape(2, 2)
    if p <= 2:
        return best_params[:2 * p].reshape(p, 2)
    return np.tile(best_params.reshape(2, 2), ((p + 1) // 2, 1))[:p]
