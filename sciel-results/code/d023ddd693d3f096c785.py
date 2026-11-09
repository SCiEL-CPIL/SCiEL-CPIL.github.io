"""Rewritten consumption-saving policy solver (vectorized backward induction)."""
import time

import numpy as np


def _crra(c, gamma):
    """CRRA utility, c floored at 1e-12. Vectorized."""
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin under the 150s kill

    beta = model["beta"]
    gamma = model["gamma"]
    R = model["R"]
    income = np.asarray(model["income"], dtype=float)
    trans = np.asarray(model["transition"], dtype=float)
    horizon = int(model["horizon"])
    mgrid = np.asarray(resource_grid, dtype=float)
    n_states = income.shape[0]
    n_m = mgrid.shape[0]

    N = 256  # consumption grid resolution (interior log-spaced)
    # interior log-spaced consumption values (relative to m)
    rel = np.exp(np.linspace(np.log(1e-3), 0.0, N))  # N values in [1e-3, 1]
    c_grid = np.concatenate([rel, [1.0]])            # N+1 values, last is m
    # per-c candidate values: (N+1, n_m)
    cvals = c_grid[:, None] * mgrid[None, :]

    # continuation values: V[z, m]
    V = np.zeros((n_states, n_m), dtype=float)

    for t in range(horizon - 1, -1, -1):
        # terminal period: consume everything
        if t == horizon - 1:
            pol = mgrid.copy()
            for z in range(n_states):
                V[z, :] = _crra(pol, gamma)
            if time.time() > deadline:
                break
            continue

        # candidate consumption for each (z, m, c): (n_states, n_m, N+1)
        cand = np.empty((n_states, n_m, cvals.shape[0]), dtype=float)
        for z in range(n_states):
            u_t = _crra(cvals, gamma)  # (N+1, n_m)
            save = mgrid[None, :] - cvals  # (N+1, n_m)
            for zp in range(n_states):
                # next resource m' = R*(m - c) + income[zp], floored at 0
                mp = np.maximum(R * save + income[zp], 0.0)  # (N+1, n_m)
                # continuation value via linear interp on mgrid
                cont = np.interp(mp, mgrid, V[zp, :])  # (N+1, n_m)
                cand[z, :, :] += trans[z, zp] * (u_t + beta * cont)

        # argmax over c (last axis)
        idx = np.argmax(cand, axis=2)  # (n_states, n_m)
        rows = np.arange(n_m)[None, :]
        pol = cvals[rows, rows, idx]  # (n_states, n_m)

        # enforce 0 < c <= m
        pol = np.clip(pol, 1e-12, mgrid[None, :])

        # terminal-period constraint: consume everything
        pol = np.where(t == horizon - 1, mgrid[None, :], pol)

        # update continuation values V[z, m] = u(c) + beta * E[V']
        for z in range(n_states):
            u_t = _crra(pol[z, :], gamma)
            for zp in range(n_states):
                mp = np.maximum(R * (mgrid - pol[z, :]) + income[zp], 0.0)
                cont = np.interp(mp, mgrid, V[zp, :])
                V[z, :] += trans[z, zp] * beta * cont
            V[z, :] = u_t + V[z, :]

        if time.time() > deadline:
            break

    # assemble result[t, z, m]
    out = np.empty((horizon, n_states, n_m), dtype=float)
    # terminal period
    for z in range(n_states):
        out[horizon - 1, z, :] = mgrid

    # recompute non-terminal policies (cheap, no V needed beyond what's stored)
    # We recompute using the same argmax logic with the final V for each t.
    # But V was overwritten; recompute policies independently via forward-free loop.
    # Recompute backward with a fresh V to get consistent policies.
    V2 = np.zeros((n_states, n_m), dtype=float)
    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            for z in range(n_states):
                V2[z, :] = _crra(mgrid, gamma)
            continue
        cand = np.empty((n_states, n_m, cvals.shape[0]), dtype=float)
        for z in range(n_states):
            u_t = _crra(cvals, gamma)
            save = mgrid[None, :] - cvals
            for zp in range(n_states):
                mp = np.maximum(R * save + income[zp], 0.0)
                cont = np.interp(mp, mgrid, V2[zp, :])
                cand[z, :, :] += trans[z, zp] * (u_t + beta * cont)
        idx = np.argmax(cand, axis=2)
        pol = cvals[rows, rows, idx]
        pol = np.clip(pol, 1e-12, mgrid[None, :])
        out[t, :, :] = pol
        for z in range(n_states):
            u_t = _crra(pol[z, :], gamma)
            for zp in range(n_states):
                mp = np.maximum(R * (mgrid - pol[z, :]) + income[zp], 0.0)
                cont = np.interp(mp, mgrid, V2[zp, :])
                V2[z, :] += trans[z, zp] * beta * cont
            V2[z, :] = u_t + V2[z, :]

    return out
