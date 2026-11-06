"""QAOA angle design via exact Walsh-transform simulation + dense global search."""
import time
import numpy as np

# Global deadline guard
_DEADLINE = time.time() + 140.0  # safety margin below 150s


def _amplitudes(n, edges, gammas, betas):
    """Compute a_w = <w| U |+>^n exactly via per-qubit state propagation.
    Returns complex array of shape (2**n,)."""
    # Precompute per-qubit rotation products for each layer.
    # After layer j: state = R_x(beta_j) R_x(gamma_j * (1 - (-1)^{w_i * s_i})) |0>
    # where s_i = XOR of neighbors' w values.
    # R_x(theta)|0> = [cos(theta/2), -i sin(theta/2)]
    # We track complex amplitude of each qubit in basis |0>,|1>.
    #
    # For each layer j and each qubit i, the effective rotation angle depends on
    # s_i = XOR of neighbors. So the qubit's final basis state (w_i) and s_i
    # determine the amplitude.
    #
    # We enumerate all w in {0,1}^n. For each w, s_i = XOR of neighbors of i.
    # Then a_w = prod_i amp_i(w_i, s_i) * exp(-i * gamma * total_phase_correction).
    #
    # Actually the C phase: exp(-i gamma * C) where C = sum_e (1 - (-1)^{w_u w_v})/2
    # = sum_e (1 - (-1)^{w_u w_v})/2. This is a global phase factor on |w>.
    #
    # Let me just simulate the full state vector for n<=10. 2^10 = 1024 states.
    # That's fine.

    # Simulate state vector |psi> of dimension 2^n.
    # Start: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    N = 1 << n
    psi = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)

    # Precompute for each layer the operator effects.
    # exp(-i beta sum_i X_i) = tensor product of R_x(beta) on each qubit.
    # R_x(beta) = [[cos(b/2), -i sin(b/2)], [-i sin(b/2), cos(b/2)]]
    #
    # exp(-i gamma C) = product over edges of exp(-i gamma (1 - (-1)^{w_u w_v})/2)
    # = product over edges of: if w_u==w_v then exp(0)=1, else exp(-i gamma).
    # So C = number of edges with w_u != w_v = cut size of w.
    # Thus exp(-i gamma C) = diag over w of exp(-i gamma * cut(w)).

    # Build cut(w) for all w.
    # cut(w) = sum over edges (u,v) of [w_u != w_v]
    cut = np.zeros(N, dtype=np.float64)
    for (u, v) in edges:
        # For each w, check if bit u != bit v.
        # bit u of w = (w >> u) & 1
        # We can vectorize: create arrays of bits.
        bits_u = (np.arange(N) >> u) & 1
        bits_v = (np.arange(N) >> v) & 1
        cut += (bits_u != bits_v).astype(np.float64)

    for j in range(len(gammas)):
        gamma = gammas[j]
        beta = betas[j]
        # Apply exp(-i gamma C): multiply psi[w] by exp(-i gamma cut(w))
        psi = psi * np.exp(-1j * gamma * cut)
        # Apply exp(-i beta sum X): tensor product of R_x(beta).
        # R_x(beta) on each qubit.
        cb = np.cos(beta / 2.0)
        sb = -1j * np.sin(beta / 2.0)
        # Apply to each qubit dimension.
        # Reshape psi to (2,)*n with qubit i as axis i (big-endian: axis 0 = MSB).
        shape = (2,) * n
        psi_r = psi.reshape(shape)
        for i in range(n):
            # Apply 2x2 matrix [[cb, sb],[sb, cb]] on axis i.
            new = np.empty_like(psi_r)
            # new[..., 0, ...] = cb*psi[...,0,...] + sb*psi[...,1,...]
            # new[..., 1, ...] = sb*psi[...,0,...] + cb*psi[...,1,...]
            sl0 = [slice(None)] * n
            sl1 = [slice(None)] * n
            sl0[i] = 0
            sl1[i] = 1
            # Extract psi[...,0,...] and psi[...,1,...]
            p0 = psi_r[tuple(sl0)]
            p1 = psi_r[tuple(sl1)]
            new[tuple(sl0)] = cb * p0 + sb * p1
            new[tuple(sl1)] = sb * p0 + cb * p1
            psi_r = new
        psi = psi_r.reshape(N)
    return psi


def _expected_cut(psi, edges):
    """Compute E[cut] = sum_e Pr[w_u != w_v] from state amplitudes psi."""
    N = len(psi)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()  # normalize for safety
    ec = 0.0
    for (u, v) in edges:
        bits_u = (np.arange(N) >> u) & 1
        bits_v = (np.arange(N) >> v) & 1
        diff = (bits_u != bits_v).astype(np.float64)
        ec += np.dot(probs, diff)
    return ec


def _exact_optimum(n, edges):
    """Brute-force MaxCut optimum for small n."""
    N = 1 << n
    best = 0
    for w in range(N):
        c = 0
        for (u, v) in edges:
            if ((w >> u) & 1) != ((w >> v) & 1):
                c += 1
        if c > best:
            best = c
    return best


def _objective(params, n, edges):
    """Return expected cut ratio for given (gamma1,beta1,gamma2,beta2)."""
    g1, b1, g2, b2 = params
    psi = _amplitudes(n, edges, [g1, g2], [b1, b2])
    return _expected_cut(psi, edges)


def _optimize_single(n, edges, deadline):
    """Dense global grid search + local refinement for one graph."""
    opt = _exact_optimum(n, edges)
    if opt == 0:
        return np.array([0.35, 0.25, 0.35, 0.25])

    # Precompute psi for a grid of (g1,b1,g2,b2)? Too big. Instead:
    # Use a two-stage approach:
    # Stage 1: coarse grid over (g1, b1, g2, b2)
    # Stage 2: local refinement with Nelder-Mead or coordinate descent.

    best_params = None
    best_val = -1.0

    # Coarse grid
    grid = np.linspace(0.0, 1.5, 8)
    # 8^4 = 4096 evaluations, each O(2^n * n) ~ 1024*10 = 10k ops. Total ~ 40M. OK.
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if time.time() > deadline - 2.0:
                        break
                    val = _objective([g1, b1, g2, b2], n, edges)
                    if val > best_val:
                        best_val = val
                        best_params = [g1, b1, g2, b2]

    # Local refinement: coordinate descent with golden section on each parameter.
    if best_params is not None and time.time() < deadline - 3.0:
        params = np.array(best_params, dtype=np.float64)
        # Coordinate descent: 5 passes, each parameter refined by golden section.
        for _pass in range(6):
            if time.time() > deadline - 2.0:
                break
            for k in range(4):
                # Golden section search on parameter k in [params[k]-0.3, params[k]+0.3]
                lo = params[k] - 0.3
                hi = params[k] + 0.3
                # Clamp to reasonable range
                lo = max(0.0, lo)
                hi = min(3.0, hi)
                if hi <= lo:
                    continue
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                # Evaluate
                p1 = params.copy(); p1[k] = x1
                p2 = params.copy(); p2[k] = x2
                v1 = _objective(p1, n, edges)
                v2 = _objective(p2, n, edges)
                for _it in range(25):
                    if time.time() > deadline - 1.5:
                        break
                    if v1 < v2:
                        lo = x1
                        x1 = x2
                        v1 = v2
                        x2 = lo + gr * (hi - lo)
                        p2 = params.copy(); p2[k] = x2
                        v2 = _objective(p2, n, edges)
                    else:
                        hi = x2
                        x2 = x1
                        v2 = v1
                        x1 = hi - gr * (hi - lo)
                        p1 = params.copy(); p1[k] = x1
                        v1 = _objective(p1, n, edges)
                # Update best
                best_x = x1 if v1 > v2 else x2
                best_v = max(v1, v2)
                if best_v > _objective(params, n, edges):
                    params[k] = best_x
        best_params = params
        best_val = _objective(best_params, n, edges)

    if best_params is None:
        best_params = np.array([0.35, 0.25, 0.35, 0.25])
    return best_params


def angles(n, edges, p):
    """Return (p, 2) array of (gamma, beta) angles."""
    global _DEADLINE
    edges = list(edges)
    # Optimize for this specific graph
    params = _optimize_single(n, edges, _DEADLINE)
    # Reshape to (p, 2): (g1, b1, g2, b2) -> [[g1,b1],[g2,b2]]
    result = np.array([[params[0], params[1]],
                       [params[2], params[3]]], dtype=np.float64)
    return result
