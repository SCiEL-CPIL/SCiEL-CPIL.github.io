"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.

Strategy:
  1. Build A (odd bonds + half field) and B (even bonds + half field).
  2. Define the 12-stage product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A).
  3. Minimize mean spectral-norm error over t = 0.25, 0.5, 1.0 via gradient descent
     with constraint projection (sum a = sum b = 1), then a high-precision polish.
  4. Deadline guard returns the best result found within the time budget.
"""
import numpy as np
import time
import sys

_DEADLINE = 140.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for a
    6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3."""
    N = 6
    delta = 0.8
    field = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1.0 + 0j]])
        for op in ops:
            result = np.kron(result, op)
        return result

    def bond_op(i, j):
        """H_ij = sx_i sx_j + sy_i sy_j + delta * sz_i sz_j (open chain, bond (i,j))."""
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        term1 = kron_all(ops)
        ops[i] = sy
        ops[j] = sy
        term2 = kron_all(ops)
        ops[i] = sz
        ops[j] = sz
        term3 = kron_all(ops)
        return term1 + term2 + delta * term3

    # Bonds for 6-site open chain: (0,1),(1,2),(2,3),(3,4),(4,5)
    # Odd bonds (even-odd): (0,1), (2,3), (4,5)
    # Even bonds (odd-even): (1,2), (3,4)
    # Each half includes half the uniform field: field/2 * sum_i sz_i
    half_field = (field / 2.0) * sum(
        kron_all([I2 * 1.0 if k != i else sz for k in range(N)])
        for i in range(N)
    )

    A = half_field.copy()
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        A = A + bond_op(i, j)

    B = half_field.copy()
    for (i, j) in [(1, 2), (3, 4)]:
        B = B + bond_op(i, j)

    return A, B


def _project(x):
    """Project x = [a_1..a_12, b_1..b_12] to satisfy sum(a)=1, sum(b)=1,
    and bounds [-50,50]. Returns projected vector and a flag for feasibility."""
    a = x[:12].copy()
    b = x[12:].copy()
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a_sum = a.sum()
    b_sum = b.sum()
    # Adjust to enforce sum=1 while staying in bounds
    a += (1.0 - a_sum) / 12.0
    b += (1.0 - b_sum) / 12.0
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    # Final fix: if clipping broke the sum, do a simple redistribution
    a += (1.0 - a.sum()) / 12.0
    b += (1.0 - b.sum()) / 12.0
    return np.concatenate([a, b])


def _compute_error(x, A, B, t_vals):
    """Compute the mean spectral-norm error of the product formula vs exact."""
    a = x[:12]
    b = x[12:]
    total_err = 0.0
    for t in t_vals:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            U = U @ (np.linalg.expm(-1j * b[k] * t * B) @ np.linalg.expm(-1j * a[k] * t * A))
        U_exact = np.linalg.expm(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(t_vals)


def _gradient(x, A, B, t_vals, eps=1e-5):
    """Finite-difference gradient of the error."""
    grad = np.zeros_like(x)
    base = _compute_error(x, A, B, t_vals)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        fp = _compute_error(_project(xp), A, B, t_vals)
        xm = x.copy()
        xm[i] -= eps
        fm = _compute_error(_project(xm), A, B, t_vals)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def coefficients(stages):
    """Optimize 12-stage product-formula coefficients for the 6-site XXZ chain."""
    # Ensure stages is 12 (pad or truncate)
    stages = 12
    A, B = _build_hamiltonians()
    t_vals = [0.25, 0.5, 1.0]

    # Initial guess: equal weights
    a0 = np.ones(12) / 12.0
    b0 = np.ones(12) / 12.0
    x = np.concatenate([a0, b0])

    best_x = x.copy()
    best_err = _compute_error(x, A, B, t_vals)

    # Gradient descent with adaptive step size
    lr = 0.05
    for it in range(300):
        if _remaining() < 15.0:
            break
        grad = _gradient(x, A, B, t_vals)
        x_new = x - lr * grad
        x_new = _project(x_new)
        err_new = _compute_error(x_new, A, B, t_vals)
        if err_new < best_err:
            best_err = err_new
            best_x = x_new.copy()
            # Adaptive learning rate: increase slightly on success
            lr = min(lr * 1.05, 0.5)
        else:
            lr = max(lr * 0.7, 0.001)
        x = x_new

    # High-precision polish: smaller steps, tighter tolerance
    if _remaining() > 10.0:
        x = best_x.copy()
        lr = 0.005
        for it in range(200):
            if _remaining() < 8.0:
                break
            grad = _gradient(x, A, B, t_vals, eps=1e-6)
            x_new = x - lr * grad
            x_new = _project(x_new)
            err_new = _compute_error(x_new, A, B, t_vals)
            if err_new < best_err:
                best_err = err_new
                best_x = x_new.copy()
                lr = min(lr * 1.02, 0.05)
            else:
                lr = max(lr * 0.8, 0.0001)
            x = x_new

    a = best_x[:12].tolist()
    b = best_x[12:].tolist()
    # Ensure exact sums
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    a_sum = sum(a)
    b_sum = sum(b)
    a = [v + (1.0 - a_sum) / 12.0 for v in a]
    b = [v + (1.0 - b_sum) / 12.0 for v in b]
    # Clip to bounds
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]
    return a, b
