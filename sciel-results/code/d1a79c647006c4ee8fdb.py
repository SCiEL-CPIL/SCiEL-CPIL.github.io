"""Optimized shallow-circuit Ising ground states via L-BFGS-B energy minimization."""
import time
import numpy as np

_DEADLINE = 150.0  # seconds; evaluator kills at this point


def _circuit_state(theta, n, layers):
    """Simulate the fixed Ry/CZ circuit and return the final state vector.

    theta: (layers, n) angles in radians.
    Start |0>^n; each layer: Ry on every qubit, then CZ on adjacent pairs
    in ascending order (no CZ after the final layer).
    Qubit i is bit i (little-endian basis indexing).
    """
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0

    for L in range(layers):
        # Ry(theta) on every qubit
        for i in range(n):
            t = theta[L, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            # Ry = [[cos, -sin],[sin, cos]] in the |0>,|1> basis
            # Apply on qubit i (bit i).
            # Iterate over basis states grouped by bit i.
            new_psi = np.empty_like(psi)
            for b in range(dim):
                bit = (b >> i) & 1
                if bit == 0:
                    # partner with bit=1
                    p1 = b | (1 << i)
                    new_psi[b] = c * psi[b] + s * psi[p1]
                    new_psi[p1] = -s * psi[b] + c * psi[p1]
                # bit==1 handled in the bit==0 iteration of its partner
            # The loop above double-counts; redo cleanly:
            new_psi = np.empty_like(psi)
            for b in range(dim):
                bit = (b >> i) & 1
                if bit == 0:
                    p1 = b | (1 << i)
                    new_psi[b] = c * psi[b] + s * psi[p1]
                    new_psi[p1] = -s * psi[b] + c * psi[p1]
            psi = new_psi

        # CZ on adjacent pairs in ascending order (skip after final layer)
        if L < layers - 1:
            for i in range(n - 1):
                # CZ flips phase of |11> on qubits i and i+1
                mask = (1 << i) | (1 << (i + 1))
                for b in range(dim):
                    if (b & mask) == mask:
                        psi[b] = -psi[b]
    return psi


def _energy_and_grad(theta, n, coupling, field, layers):
    """Compute energy E = <psi|H|psi>/n and its gradient w.r.t. theta.

    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i (open chain).
    Returns (energy_per_spin, grad_per_spin) where grad has shape (layers, n).
    """
    dim = 1 << n
    psi = _circuit_state(theta, n, layers)

    # Compute expectation values.
    # Z_i Z_{i+1}: diagonal in computational basis.
    zz = 0.0
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zz_val = (1 if bi == 0 else -1) * (1 if bj == 0 else -1)
            zz += zz_val * (psi[b].real * psi[b].real + psi[b].imag * psi[b].imag)

    # X_i: off-diagonal.
    xx = 0.0
    for i in range(n):
        for b in range(dim):
            bit = (b >> i) & 1
            p = b ^ (1 << i)
            # <psi| X_i |psi> = 2 Re( conj(psi[b]) psi[p] ) summed over b with bit=0
            if bit == 0:
                xx += 2.0 * (psi[b].real * psi[p].real + psi[b].imag * psi[p].imag)

    energy_total = -coupling * zz - field * xx
    energy = energy_total / n

    # Gradient: finite-difference is slow; use analytic via parameter-shift-like.
    # For efficiency and correctness, use central finite differences with a
    # small step, but that's O(layers*n) simulations. Instead, compute via
    # automatic approach: re-simulate with perturbed angles.
    # To keep it fast, use a modest step and forward differences.
    step = 1e-4
    grad = np.zeros_like(theta)
    base = energy
    for L in range(layers):
        for i in range(n):
            old = theta[L, i]
            theta[L, i] = old + step
            psi_p = _circuit_state(theta, n, layers)
            zz_p = 0.0
            for j in range(n - 1):
                for b in range(dim):
                    bi = (b >> j) & 1
                    bj = (b >> (j + 1)) & 1
                    zz_val = (1 if bi == 0 else -1) * (1 if bj == 0 else -1)
                    zz_p += zz_val * (psi_p[b].real * psi_p[b].real + psi_p[b].imag * psi_p[b].imag)
            xx_p = 0.0
            for j in range(n):
                for b in range(dim):
                    bit = (b >> j) & 1
                    p = b ^ (1 << j)
                    if bit == 0:
                        xx_p += 2.0 * (psi_p[b].real * psi_p[p].real + psi_p[b].imag * psi_p[p].imag)
            e_p = (-coupling * zz_p - field * xx_p) / n
            theta[L, i] = old
            grad[L, i] = (e_p - base) / step

    return energy, grad


def _objective_and_grad(x, n, coupling, field, layers):
    """Flatten theta for L-BFGS-B."""
    theta = x.reshape(layers, n)
    e, g = _energy_and_grad(theta, n, coupling, field, layers)
    return e, g.reshape(-1)


def angles(n, coupling, field, layers):
    start = time.time()
    deadline = start + _DEADLINE - 5.0  # safety margin

    # Initial guess: start from the seed (|+> on last layer) but optimize all.
    theta = np.zeros((layers, n))
    theta[-1] = np.pi / 2.0

    best_theta = theta.copy()
    best_val = np.inf

    # Evaluate initial.
    try:
        e0, _ = _objective_and_grad(theta.ravel(), n, coupling, field, layers)
        best_val = e0
    except Exception:
        pass

    # Try a few random restarts and L-BFGS-B, guarded by deadline.
    rng = np.random.default_rng(12345)
    restarts = 0
    max_restarts = 6

    while time.time() < deadline and restarts < max_restarts:
        # Current theta (first restart uses seed, others random).
        if restarts == 0:
            x0 = best_theta.ravel().copy()
        else:
            x0 = best_theta.ravel().copy() + rng.normal(0, 0.5, size=layers * n)
            # Wrap into a reasonable range
            x0 = np.clip(x0, -np.pi, np.pi)

        def fun(x):
            e, _ = _objective_and_grad(x, n, coupling, field, layers)
            return e

        def fun_and_grad(x):
            e, g = _objective_and_grad(x, n, coupling, field, layers)
            return e, g

        try:
            from scipy.optimize import minimize
            res = minimize(
                fun_and_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 40, "maxfun": 200, "gtol": 1e-6},
            )
            if res.success or (res.fun is not None and res.fun < best_val):
                cand = res.x.reshape(layers, n)
                cand_val = res.fun
                if cand_val < best_val:
                    best_val = cand_val
                    best_theta = cand.copy()
        except Exception:
            pass

        restarts += 1
        if time.time() > deadline:
            break

    return best_theta
