"""Combined program: vectorized independent-optimization DP + vectorized joint refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _build_grids(k, n_coarse, n_fine):
    """Depth grid focused around 1/k where the optimum lives."""
    lo = 1e-4
    hi = max(10.0 / k, 2.0)
    # Coarse grid over full range
    grid_coarse = np.linspace(lo, hi, n_coarse)
    # Fine grid around the critical region [lo, 3/k]
    grid_fine = np.linspace(lo, min(3.0 / k, hi), n_fine)
    grid = np.unique(np.concatenate([grid_coarse, grid_fine]))
    grid = np.clip(grid, lo, hi)
    return grid


def _refine_depths(opt_bid, opt_ask, sigma, A, k, gamma, T, steps, Q, time_budget=5.0):
    """
    Local refinement pass: for each (n, q), try small perturbations around the
    current grid depth to see if a slightly different depth gives better value.
    Uses the value function from the joint DP as a surrogate.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    n_steps = steps

    # We need the value function V[n, q] from the joint DP.
    # Recompute it quickly using the joint grid.
    jn = 36
    joint_grid = np.linspace(1e-4, min(3.0 / k, 10.0 / k), jn)
    joint_probs = 1.0 - np.exp(-A * np.exp(-k * joint_grid) * dt)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    V = np.zeros((n_steps + 1, n_q))
    # V[n+1, q] is the value-to-go from step n+1
    # We compute backward: V[n, q] = max over actions of immediate + V[n+1, q']
    
    # Rebuild V from the joint DP (same as stage 2)
    for n in range(n_steps - 1, -1, -1):
        V_curr = np.zeros(n_q)
        if n_q > 2:
            idx_int = np.arange(1, n_q - 1)
            Vq = V[n + 1, idx_int]
            Vp = V[n + 1, idx_int + 1]
            Vm = V[n + 1, idx_int - 1]
            dp = Vp - Vq
            dm = Vm - Vq
            cross = Vp - 2.0 * Vq + Vm

            pb = joint_probs[None, :, None]
            pa = joint_probs[None, None, :]
            db = joint_grid[None, :, None]
            da = joint_grid[None, None, :]

            next_val = (Vq[:, None, None] +
                        pb * dp[:, None, None] +
                        pa * dm[:, None, None] -
                        pb * pa * cross[:, None, None])
            reward = db * pb + da * pa
            total = reward - penalties[idx_int][:, None, None] + next_val

            flat = total.reshape(len(idx_int), -1)
            best_flat = np.argmax(flat, axis=1)
            V_curr[idx_int] = flat[np.arange(len(idx_int)), best_flat]

        if Q > 0:
            idx_q = 2 * Q
            delta = V[n + 1, idx_q - 1] - V[n + 1, idx_q]
            vals = joint_probs * (joint_grid + delta)
            bi = int(np.argmax(vals))
            V_curr[idx_q] = vals[bi] - penalties[idx_q] + V[n + 1, idx_q]

            idx_q = 0
            delta = V[n + 1, idx_q + 1] - V[n + 1, idx_q]
            vals = joint_probs * (joint_grid + delta)
            bi = int(np.argmax(vals))
            V_curr[idx_q] = vals[bi] - penalties[idx_q] + V[n + 1, idx_q]

        V[n] = V_curr

    # Now refine: for each (n, q), try a few candidate depths near the current one
    # and pick the best if it improves the one-step value.
    # We'll do this in chunks to respect time budget.
    start_time = time.time()
    n_candidates = 5
    candidate_offsets = np.array([-2, -1, 0, 1, 2]) * (1e-4 / k)

    for n in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        for q_idx in range(n_q):
            if time.time() - start_time > time_budget:
                break

            # Get current depths
            db_curr = opt_bid[n, q_idx]
            da_curr = opt_ask[n, q_idx]

            # For each side, try candidate depths
            # Compute one-step value for each candidate
            # V_next_q_plus = V[n+1, q_idx+1] if q_idx < n_q-1
            # V_next_q_minus = V[n+1, q_idx-1] if q_idx > 0

            Vq = V[n + 1, q_idx]
            if q_idx < n_q - 1:
                Vp = V[n + 1, q_idx + 1]
            else:
                Vp = None
            if q_idx > 0:
                Vm = V[n + 1, q_idx - 1]
            else:
                Vm = None

            # Try bid refinements
            if db_curr < np.inf:
                db_candidates = db_curr + candidate_offsets
                db_candidates = np.clip(db_candidates, 1e-4, 10.0 / k)
                db_probs = 1.0 - np.exp(-A * np.exp(-k * db_candidates) * dt)
                if Vp is not None:
                    dp = Vp - Vq
                    bid_vals = db_probs * (db_candidates + dp)
                    best_bi = int(np.argmax(bid_vals))
                    opt_bid[n, q_idx] = db_candidates[best_bi]
                else:
                    # At q = Q, no bid fill, so no refinement needed
                    pass

            # Try ask refinements
            if da_curr < np.inf:
                da_candidates = da_curr + candidate_offsets
                da_candidates = np.clip(da_candidates, 1e-4, 10.0 / k)
                da_probs = 1.0 - np.exp(-A * np.exp(-k * da_candidates) * dt)
                if Vm is not None:
                    dm = Vm - Vq
                    ask_vals = da_probs * (da_candidates + dm)
                    best_ai = int(np.argmax(ask_vals))
                    opt_ask[n, q_idx] = da_candidates[best_ai]
                else:
                    # At q = -Q, no ask fill, so no refinement needed
                    pass

    return opt_bid, opt_ask


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Two-stage DP over (time, inventory).
    Stage 1: independent bid/ask optimization (vectorized over inventory).
    Stage 2: joint (db, da) grid refinement (vectorized over inventory).
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    n_coarse, n_fine = 40, 80
    depth_grid = _build_grids(k, n_coarse, n_fine)
    n_d = len(depth_grid)

    # Joint refinement grid (finer, focused)
    jn = 36
    joint_grid = np.linspace(1e-4, min(3.0 / k, 10.0 / k), jn)

    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    joint_probs = 1.0 - np.exp(-A * np.exp(-k * joint_grid) * dt)

    V_next = np.zeros(2 * Q + 1)
    q_vals = np.arange(-Q, Q + 1)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)

    # Stage 1: fast independent optimization, vectorized over inventory
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 10.0:
            break

        V_curr = np.zeros(2 * Q + 1)
        opt_bid_n = np.full(2 * Q + 1, np.inf)
        opt_ask_n = np.full(2 * Q + 1, np.inf)

        # Interior: q in [-Q+1, Q-1]
        if 2 * Q + 1 > 2:
            idx_int = np.arange(1, 2 * Q)  # indices 1..2Q-1
            q_int = idx_int - Q  # actual inventory values

            dp_int = V_next[idx_int + 1] - V_next[idx_int]  # delta_plus
            dm_int = V_next[idx_int - 1] - V_next[idx_int]  # delta_minus

            # vals_bid shape: (n_int, n_d)
            vals_bid = fill_probs[None, :] * (depth_grid[None, :] + dp_int[:, None])
            vals_ask = fill_probs[None, :] * (depth_grid[None, :] + dm_int[:, None])

            best_bi = np.argmax(vals_bid, axis=1)
            best_ai = np.argmax(vals_ask, axis=1)

            V_curr[idx_int] = vals_bid[np.arange(len(idx_int)), best_bi] + \
                              vals_ask[np.arange(len(idx_int)), best_ai] - \
                              penalties[idx_int] + V_next[idx_int]
            opt_bid_n[idx_int] = depth_grid[best_bi]
            opt_ask_n[idx_int] = depth_grid[best_ai]

        # q = Q (index 2Q): only ask
        if Q > 0:
            idx_q = 2 * Q
            delta = V_next[idx_q - 1] - V_next[idx_q]
            vals = fill_probs * (depth_grid + delta)
            bi = int(np.argmax(vals))
            V_curr[idx_q] = vals[bi] - penalties[idx_q] + V_next[idx_q]
            opt_ask_n[idx_q] = depth_grid[bi]
            # opt_bid_n[idx_q] stays inf

        # q = -Q (index 0): only bid
        if Q > 0:
            idx_q = 0
            delta = V_next[idx_q + 1] - V_next[idx_q]
            vals = fill_probs * (depth_grid + delta)
            bi = int(np.argmax(vals))
            V_curr[idx_q] = vals[bi] - penalties[idx_q] + V_next[idx_q]
            opt_bid_n[idx_q] = depth_grid[bi]
            # opt_ask_n[idx_q] stays inf

        # Q = 0 case: only q=0, which is interior (handled above if 2Q+1>2 fails)
        if Q == 0:
            # Only one state, no fills possible effectively
            V_curr[0] = -penalties[0] + V_next[0]
            # No quotes needed

        opt_bid[n] = opt_bid_n
        opt_ask[n] = opt_ask_n
        V_next = V_curr

    # Stage 2: joint refinement from the end, while time permits
    # Re-seed V_next to zero and recompute backward with joint grid
    V_next = np.zeros(2 * Q + 1)
    remaining_steps = steps
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 15.0:
            break
        remaining_steps = n + 1

        V_curr = np.zeros(2 * Q + 1)
        opt_bid_n = np.full(2 * Q + 1, np.inf)
        opt_ask_n = np.full(2 * Q + 1, np.inf)

        # Interior: q in [-Q+1, Q-1]
        if 2 * Q + 1 > 2:
            idx_int = np.arange(1, 2 * Q)
            Vq = V_next[idx_int]
            Vp = V_next[idx_int + 1]
            Vm = V_next[idx_int - 1]

            dp = Vp - Vq  # shape (n_int,)
            dm = Vm - Vq  # shape (n_int,)
            cross = Vp - 2.0 * Vq + Vm  # shape (n_int,)

            # Vectorized over inventory: build 3D array (n_int, jn, jn)
            # pb shape: (1, jn, 1), pa shape: (1, 1, jn)
            # db shape: (1, jn, 1), da shape: (1, 1, jn)
            pb = joint_probs[None, :, None]  # (1, jn, 1)
            pa = joint_probs[None, None, :]  # (1, 1, jn)
            db = joint_grid[None, :, None]   # (1, jn, 1)
            da = joint_grid[None, None, :]   # (1, 1, jn)

            # next_val: (n_int, jn, jn)
            next_val = (Vq[:, None, None] +
                        pb * dp[:, None, None] +
                        pa * dm[:, None, None] -
                        pb * pa * cross[:, None, None])

            # reward: (n_int, jn, jn)
            reward = db * pb + da * pa  # (1, jn, jn) broadcast to (n_int, jn, jn)

            total = reward - penalties[idx_int][:, None, None] + next_val

            # Find best (bi, ai) for each inventory
            flat = total.reshape(len(idx_int), -1)
            best_flat = np.argmax(flat, axis=1)
            best_bi = best_flat // jn
            best_ai = best_flat % jn

            V_curr[idx_int] = flat[np.arange(len(idx_int)), best_flat]
            opt_bid_n[idx_int] = joint_grid[best_bi]
            opt_ask_n[idx_int] = joint_grid[best_ai]

        # q = Q (index 2Q): only ask
        if Q > 0:
            idx_q = 2 * Q
            delta = V_next[idx_q - 1] - V_next[idx_q]
            vals = joint_probs * (joint_grid + delta)
            bi = int(np.argmax(vals))
            V_curr[idx_q] = vals[bi] - penalties[idx_q] + V_next[idx_q]
            opt_ask_n[idx_q] = joint_grid[bi]

        # q = -Q (index 0): only bid
        if Q > 0:
            idx_q = 0
            delta = V_next[idx_q + 1] - V_next[idx_q]
            vals = joint_probs * (joint_grid + delta)
            bi = int(np.argmax(vals))
            V_curr[idx_q] = vals[bi] - penalties[idx_q] + V_next[idx_q]
            opt_bid_n[idx_q] = joint_grid[bi]

        if Q == 0:
            V_curr[0] = -penalties[0] + V_next[0]

        opt_bid[n] = opt_bid_n
        opt_ask[n] = opt_ask_n
        V_next = V_curr

    # Stage 3: local depth refinement if time permits
    if time.time() < _DEADLINE - 8.0:
        try:
            opt_bid, opt_ask = _refine_depths(
                opt_bid, opt_ask, sigma, A, k, gamma, T, steps, Q,
                time_budget=min(6.0, _DEADLINE - time.time() - 8.0)
            )
        except Exception:
            pass

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """Avellaneda-Stoikov style analytical fallback."""
    sigma = params["sigma"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    dt = T / steps
    remaining = T - t
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    bid_depths = np.clip(bid_depths, 1e-4, 10.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 10.0 / k)
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 12.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        # Ensure valid depths: replace values <= 0 or NaN with inf
        bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 1e-4), bid_depths, np.inf)
        ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 1e-4), ask_depths, np.inf)
        return (np.asarray(bid_depths, dtype=np.float64),
                np.asarray(ask_depths, dtype=np.float64))
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
