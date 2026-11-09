"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 125.0  # Safety margin for 120s/150s limits


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Return (A, B): odd-bond+half-field and even-bond+half-field for 6-site open XXZ."""
    N = 6
    delta = 0.8
    field = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    # Pre-compute single-site operators in full space
    Sx_list, Sy_list, Sz_list = [], [], []
    for i in range(N):
        ops_x = [I2] * N
        ops_y = [I2] * N
        ops_z = [I2] * N
        ops_x[i] = sx
        ops_y[i] = sy
        ops_z[i] = sz
        Sx_list.append(kron_all(ops_x))
        Sy_list.append(kron_all(ops_y))
        Sz_list.append(kron_all(ops_z))

    A = np.zeros((2 ** N, 2 ** N), dtype=complex)
    B = np.zeros((2 ** N, 2 ** N), dtype=complex)
    
    # Bonds: 0-1 (A), 1-2 (B), 2-3 (A), 3-4 (B), 4-5 (A)
    for i in range(N - 1):
        J = Sx_list[i] @ Sx_list[i + 1] + Sy_list[i] @ Sy_list[i + 1] + delta * Sz_list[i] @ Sz_list[i + 1]
        if i % 2 == 0:
            A += J
        else:
            B += J
            
    # Field: split evenly. Site 0,2,4 to A; Site 1,3,5 to B.
    for i in range(N):
        h = field * Sz_list[i] * 0.5
        if i % 2 == 0:
            A += h
        else:
            B += h
    return A, B


class _ExpHelper:
    """Fast exponential calculation using pre-computed eigen-decomposition."""
    def __init__(self, M):
        self.M = M
        # M is Hermitian, so we can use eigh for stability and speed
        self.eigvals, self.eigvecs = np.linalg.eigh(M)
        self.inv_eigvecs = np.linalg.inv(self.eigvecs)
        
    def exp(self, s):
        """Compute exp(-i * s * M)"""
        # exp(-i s M) = V exp(-i s D) V^-1
        # D is diagonal with eigvals
        phase = np.exp(-1j * s * self.eigvals)
        # (V * phase) @ V^-1
        # We can optimize this as V @ diag(phase) @ V^-1
        # Which is equivalent to (V * phase) @ V_inv if we broadcast correctly?
        # V is (N, N), phase is (N,). V * phase scales columns.
        # So (V * phase) @ self.inv_eigvecs
        return (self.eigvecs * phase) @ self.inv_eigvecs


def coefficients(stages):
    global _T0
    stages = int(stages)
    if stages < 2:
        stages = 2
    
    tvals = [0.25, 0.5, 1.0]

    A, B = _build_hamiltonians()
    AB = A + B
    
    # Pre-compute helpers
    helper_A = _ExpHelper(A)
    helper_B = _ExpHelper(B)
    helper_AB = _ExpHelper(AB)
    
    # Pre-compute exact propagators for t in tvals
    exact_prop = {t: helper_AB.exp(t) for t in tvals}

    # Initial guess: uniform
    a = [1.0 / stages] * stages
    b = [1.0 / stages] * stages

    def score(a, b):
        err_sum = 0.0
        for t in tvals:
            U = np.eye(A.shape[0], dtype=complex)
            for k in range(stages):
                bk = b[k]
                if abs(bk) > 1e-15:
                    U = U @ helper_B.exp(bk * t)
                ak = a[k]
                if abs(ak) > 1e-15:
                    U = U @ helper_A.exp(ak * t)
            err_sum += np.linalg.norm(U - exact_prop[t])
        return err_sum / len(tvals)

    best_a, best_b, best_s = a[:], b[:], score(a, b)

    rng = np.random.default_rng(42)
    
    # Optimization Loop
    # We use a simple random search with constraint projection.
    # Since 64x64 matrix mult is fast, we can afford many iterations if we are careful.
    while _remaining() > 5.0:
        # Pick a random vector to perturb (a or b)
        which = rng.integers(0, 2)
        cur_list = best_a if which == 0 else best_b
        other_list = best_b if which == 0 else best_a
        
        # Generate a random perturbation
        # We perturb a few random indices to allow for more complex structures
        n_perturb = int(rng.integers(1, min(4, stages)))
        indices = rng.choice(stages, size=n_perturb, replace=False)
        
        improved = False
        for _ in range(5): # Try 5 random steps
            trial = cur_list[:]
            for idx in indices:
                # Step size depends on current value? No, fixed small step.
                # Symmetric step sizes
                step = float(rng.normal(0, 0.05))
                trial[idx] += step
            
            # Project to sum=1
            s_diff = 1.0 - sum(trial)
            if abs(s_diff) > 1e-12:
                # Distribute error evenly
                for j in range(stages):
                    trial[j] += s_diff / stages
            
            # Clamp to [-50, 50]
            # Note: Clamping might break the sum again, but for small steps it's rare.
            # If it happens, we re-project.
            clamped = False
            for j in range(stages):
                if trial[j] < -50.0:
                    trial[j] = -50.0
                    clamped = True
                elif trial[j] > 50.0:
                    trial[j] = 50.0
                    clamped = True
            if clamped:
                s_diff = 1.0 - sum(trial)
                for j in range(stages):
                    trial[j] += s_diff / stages
                # Re-clamp if necessary (unlikely to oscillate)
                for j in range(stages):
                    trial[j] = min(50.0, max(-50.0, trial[j]))
                # Final sum fix
                s_diff = 1.0 - sum(trial)
                for j in range(stages):
                    trial[j] += s_diff / stages

            s = score(trial, other_list)
            if s < best_s - 1e-9:
                if which == 0:
                    best_a, best_s = trial[:], s
                else:
                    best_b, best_s = trial[:], s
                improved = True
                break
        
        # If no improvement in random walk, try a coordinate descent step on a random single index
        if not improved and _remaining() > 5.0:
            idx = int(rng.integers(0, stages))
            for step in [0.01, -0.01, 0.005, -0.005]:
                trial = cur_list[:]
                trial[idx] += step
                s_diff = 1.0 - sum(trial)
                # Correct one other random index to maintain sum (faster than spreading)
                # Or just spread. Spreading is safer for symmetry.
                for j in range(stages):
                    trial[j] += s_diff / stages
                s = score(trial, other_list)
                if s < best_s - 1e-9:
                    if which == 0:
                        best_a, best_s = trial[:], s
                    else:
                        best_b, best_s = trial[:], s
                    break

    # Final cleanup
    for lst in (best_a, best_b):
        s = sum(lst)
        if abs(s - 1.0) > 1e-12:
            for j in range(stages):
                lst[j] += (1.0 - s) / stages
        for j in range(stages):
            lst[j] = min(50.0, max(-50.0, lst[j]))
        # Re-fix sum after clamp
        s = sum(lst)
        if abs(s - 1.0) > 1e-12:
            for j in range(stages):
                lst[j] += (1.0 - s) / stages

    return best_a, best_b
