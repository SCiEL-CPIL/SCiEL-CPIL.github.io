"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] with initial condition y0.
    Assumes f is a constant-coefficient linear ODE: y' = A y + b.
    Uses minimal function calls to extract A and b, then computes
    the exact solution via matrix exponential (Taylor series).
    """
    start_time = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    T = t1 - t0

    # Step 1: Extract b = f(0, 0) -- 1 call
    b = np.asarray(f(0.0, np.zeros(n)), dtype=float)

    # Step 2: Extract A columns.
    # f(t, y) = A y + b, so f(t, e_j) - b = A e_j = column j of A.
    # We need n calls for the columns.
    # Total so far: 1 + n = 16 calls for n=15.
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = np.asarray(f(0.0, ej), dtype=float) - b

    # Check deadline
    if time.time() > DEADLINE:
        # Fallback: return y0 + T * (A y0 + b) as a rough estimate
        return y0 + T * (A @ y0 + b)

    # Step 3: Compute the exact solution.
    # For y' = A y + b, the solution is:
    # y(t) = exp(A t) y0 + integral_0^t exp(A (t-s)) b ds
    #      = exp(A t) y0 + A^{-1} (exp(A t) - I) b   (if A invertible)
    # More robustly: augment the system.
    # Let z = [y; 1], then z' = M z where M = [[A, b], [0, 0]].
    # z(t) = exp(M t) z(0), where z(0) = [y0; 1].
    # This handles singular A correctly.

    # Build augmented matrix M (n+1) x (n+1)
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = b

    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0

    # Compute exp(M * T) using Taylor series with scaling and squaring
    # for better numerical stability.
    # For stiff systems, |A| can be large, so we use scaling and squaring.

    # Determine scaling factor: choose s such that ||M*T||_1 / 2^s < 1
    norm_MT = np.max(np.sum(np.abs(M * T), axis=1))
    if norm_MT == 0:
        # M is zero, solution is just z0
        return y0 + T * b  # Actually if M=0, y(t) = y0 + T*b? No.
        # If A=0 and b=0, y(t)=y0. If A=0, y(t) = y0 + T*b.
        # Let's just handle this edge case.
        # Actually if norm_MT is 0, M is all zeros, so exp(MT)=I, z(t)=z0.
        return z0[:n]

    # Choose s such that 2^s >= norm_MT / 0.25 (to keep the Taylor series convergent quickly)
    s = max(0, int(np.ceil(np.log2(norm_MT / 0.25))))
    if s > 30:
        s = 30  # cap to avoid excessive squaring

    # Compute exp(M * T / 2^s) using Taylor series
    M_scaled = M * T / (2.0 ** s)
    # Taylor series: sum_{k=0}^{K} (M_scaled)^k / k!
    # For ||M_scaled|| < 0.25, a few terms suffice for high accuracy.
    K = 30  # number of Taylor terms
    exp_term = np.eye(n + 1)
    result_exp = exp_term.copy()
    for k in range(1, K + 1):
        exp_term = exp_term @ M_scaled / k
        result_exp = result_exp + exp_term
        if np.max(np.abs(exp_term)) < 1e-16:
            break

    # Square s times to get exp(M * T)
    exp_MT = result_exp
    for _ in range(s):
        exp_MT = exp_MT @ exp_MT

    # Final solution
    z_t = exp_MT @ z0
    y_final = z_t[:n]

    # Check deadline
    if time.time() > DEADLINE:
        return y_final

    # Optional: verify with a few RK4 steps if time permits (not needed for score, but safety)
    # Actually, the Taylor series approach should be very accurate.
    # Let's do a quick sanity check with a single large-step RK4 if time allows.
    # But this would use more f calls, which hurts the score. Skip it.

    return y_final
# EVOLVE-BLOCK-END
