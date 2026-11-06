"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula using a 4th-order symmetric splitting scheme.
    We use the Yoshida 4th-order coefficients applied in a way that fits the
    12-stage (B, A) * 6 structure, or more directly, we use a known
    4th-order symmetric method.
    
    The structure is U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A).
    
    A good starting point is the 4th-order Suzuki-Trotter (Yoshida) coefficients
    mapped to this format. However, a simpler and often effective approach for
    12 stages is to use a 4th-order method with 3 substeps or similar.
    
    Let's use the 4th-order coefficients from the current program (Yoshida)
    but ensure they are correctly mapped to the 12-stage format.
    
    The current program uses 2 substeps of Yoshida 4th order.
    Let's verify the sums and structure.
    
    Alternative: Use a 4th-order method with 4 substeps? No, 12 stages / 2 per substep = 6 substeps.
    If we use 6 substeps of 2nd order Strang, we get 2nd order global.
    If we use 4th order coefficients, we need to map them.
    
    Let's try to optimize the coefficients using a simple local search if time permits,
    starting from a good initial guess (Yoshida 4th order).
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safe margin before 150s limit

    # Yoshida 4th order coefficients
    two_thirds_root = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - two_thirds_root
    c1 = 1.0 / denom
    c2 = -two_thirds_root / denom
    c3 = c1

    # Base 4th order scheme for one step (size dt):
    # exp(-i c2 dt/2 B) exp(-i c2 dt A) exp(-i (c2+c1) dt/2 B) exp(-i c1 dt A)
    # exp(-i (c1+c3) dt/2 B) exp(-i c3 dt A) exp(-i c3 dt/2 B)
    # This is 7 B/A terms, but we have pairs (B_k, A_k).
    # We can merge adjacent B terms.
    
    # Let's just use the coefficients from the current program as a starting point,
    # which was already a valid 4th-order scheme.
    b = [0.0] * stages
    a = [0.0] * stages
    
    # From current program (2 substeps of Yoshida):
    b[0] = c2 / 4.0
    a[0] = c2 / 2.0
    b[1] = (c2 + c1) / 4.0
    a[1] = c1 / 2.0
    b[2] = (c1 + c3) / 4.0
    a[2] = c3 / 2.0
    b[3] = (c3 + c2) / 4.0
    a[3] = c2 / 2.0
    b[4] = (c2 + c1) / 4.0
    a[4] = c1 / 2.0
    b[5] = (c1 + c3) / 4.0
    a[5] = c3 / 2.0
    b[6] = c3 / 4.0
    a[6] = 0.0

    # Check sums
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1.0) > 1e-10 or abs(sum_b - 1.0) > 1e-10:
        # Fallback to simple Strang
        substeps = stages // 2
        a = [0.5 / substeps] * stages
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(b) < stages:
            b.append(0.0)
        return a, b

    # Try to improve with local optimization if time permits
    if time.time() < deadline - 60:
        try:
            # Build Hamiltonians
            N = 6
            dim = 2 ** N
            delta = 0.8
            field = 0.3

            sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
            sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
            sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)

            def tensor_op(op, site):
                result = np.array([[1.0]], dtype=complex)
                for i in range(N):
                    if i == site:
                        result = np.kron(result, op)
                    else:
                        result = np.kron(result, np.eye(2, dtype=complex))
                return result

            A = np.zeros((dim, dim), dtype=complex)
            B = np.zeros((dim, dim), dtype=complex)

            def bond_term(j, j1):
                sz_j = tensor_op(sz, j)
                sz_j1 = tensor_op(sz, j1)
                sx_j = tensor_op(sx, j)
                sx_j1 = tensor_op(sx, j1)
                sy_j = tensor_op(sy, j)
                sy_j1 = tensor_op(sy, j1)
                return (delta / 4.0) * (sz_j @ sz_j1) + (1 / 4.0) * (sx_j @ sx_j1 + sy_j @ sy_j1)

            for j, j1 in [(0, 1), (2, 3), (4, 5)]:
                A += bond_term(j, j1)
            for j, j1 in [(1, 2), (3, 4)]:
                B += bond_term(j, j1)
            for site in range(N):
                A += field / 4.0 * tensor_op(sz, site)
                B += field / 4.0 * tensor_op(sz, site)

            H = A + B

            def compute_error(a_coeffs, b_coeffs, t):
                U = np.eye(dim, dtype=complex)
                for k in range(len(a_coeffs)):
                    U = np.linalg.matrix_exp(-1j * b_coeffs[k] * t * B) @ U
                    U = np.linalg.matrix_exp(-1j * a_coeffs[k] * t * A) @ U
                U_exact = np.linalg.matrix_exp(-1j * t * H)
                return np.linalg.norm(U - U_exact, 2)

            def total_error(a_coeffs, b_coeffs):
                errors = [compute_error(a_coeffs, b_coeffs, t) for t in [0.25, 0.5, 1.0]]
                return sum(errors) / 3.0

            best_a = a[:]
            best_b = b[:]
            best_err = total_error(best_a, best_b)

            # Simple coordinate descent
            step = 0.01
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                improved = False
                for k in range(stages):
                    for delta in [-step, step]:
                        trial_a = best_a[:]
                        trial_a[k] += delta
                        s = sum(trial_a)
                        if abs(s - 1.0) > 1e-10:
                            trial_a = [x / s for x in trial_a]
                        if all(-50 <= x <= 50 for x in trial_a):
                            err = total_error(trial_a, best_b)
                            if err < best_err - 1e-12:
                                best_a = trial_a
                                best_err = err
                                improved = True

                    for delta in [-step, step]:
                        trial_b = best_b[:]
                        trial_b[k] += delta
                        s = sum(trial_b)
                        if abs(s - 1.0) > 1e-10:
                            trial_b = [x / s for x in trial_b]
                        if all(-50 <= x <= 50 for x in trial_b):
                            err = total_error(best_a, trial_b)
                            if err < best_err - 1e-12:
                                best_b = trial_b
                                best_err = err
                                improved = True

                if not improved:
                    step /= 2.0
                    if step < 1e-4:
                        break

            a = best_a
            b = best_b

        except Exception:
            pass

    return a, b
# EVOLVE-BLOCK-END
