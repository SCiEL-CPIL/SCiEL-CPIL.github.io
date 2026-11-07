"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for optimal policy lookups
_policy_cache = {}
_START_TIME = time.time()


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q, n_depth_candidates=80, max_depth=None):
    """
    Compute the optimal bid/ask depths for each inventory level using DP.
    Returns arrays bid_depths_by_q, ask_depths_by_q of shape (2*Q+1,) each.
    """
    if max_depth is None:
        max_depth = 8.0 / k

    # Precompute depth grid (log-spaced for better resolution near 0)
    # We need depths in (0, max_depth]
    # Use log-spaced grid
    depth_grid = np.logspace(np.log10(1e-6), np.log10(max_depth), n_depth_candidates)
    n_d = len(depth_grid)

    dt = T / steps

    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)

    # Precompute depth * p_fill for reward
    depth_pfill = depth_grid * p_fill

    # Inventory levels: -Q to Q
    q_levels = np.arange(-Q, Q + 1)
    n_q = len(q_levels)

    # DP: V[q] = optimal expected reward from state q with remaining steps
    # At each step, we choose bid_depth and ask_depth to maximize:
    #   reward = bid_depth * p_fill_bid + ask_depth * p_fill_ask - (gamma/2)*sigma^2*q^2*dt
    #   + E[V[next_q]]
    # where next_q = q + 1 if bid hit, q - 1 if ask hit, q otherwise (with appropriate probabilities)
    # Actually: bid hit -> q increases by 1 (we bought), ask hit -> q decreases by 1 (we sold)
    # Wait, let me re-read: "A fill earns its depth and moves inventory one unit"
    # "no bid fill at q = Q, no ask fill at q = -Q"
    # So bid fill: we buy, inventory increases (q -> q+1), constraint q < Q
    # Ask fill: we sell, inventory decreases (q -> q-1), constraint q > -Q

    # The mid moves randomly but that doesn't affect inventory or the depth choice directly
    # (depths are relative to mid, and the objective only depends on depths earned and inventory penalty)

    # So the DP is:
    # V_n(q) = max_{b,a} [ b*p_fill(b) * I(q<Q) + a*p_fill(a) * I(q>-Q) - (gamma/2)*sigma^2*q^2*dt
    #            + (1-p_b)(1-p_a)*V_{n+1}(q) + p_b*(1-p_a)*V_{n+1}(q+1) + (1-p_b)*p_a*V_{n+1}(q-1) + p_b*p_a*V_{n+1}(q) ]
    # where p_b = p_fill(b) if q<Q else 0, p_a = p_fill(a) if q>-Q else 0

    # Terminal: V_T(q) = 0 (no terminal charge)

    # We can vectorize over q for each step, and for each q, optimize over (b,a) grid.
    # But that's n_q * n_d^2 per step * steps = 61 * 6400 * 1000 = 390M operations, too slow.

    # Better: separate the optimization. Note that the objective separates in b and a:
    # max_b [ b*p_fill(b) * I(q<Q) + p_b * (V(q+1) - V(q)) * I(q<Q) ]  (if q<Q)
    # max_a [ a*p_fill(a) * I(q>-Q) + p_a * (V(q-1) - V(q)) * I(q>-Q) ]  (if q>-Q)
    # plus the base term - (gamma/2)*sigma^2*q^2*dt + (1-p_b)(1-p_a)*V(q) ... 

    # Actually let me expand more carefully:
    # Let pb = p_fill(b) if q<Q else 0
    # Let pa = p_fill(a) if q>-Q else 0
    # Objective = b*pb + a*pa - c*q^2 + (1-pb)(1-pa)V(q) + pb(1-pa)V(q+1) + (1-pb)pa*V(q-1) + pb*pa*V(q)
    # where c = (gamma/2)*sigma^2*dt, and V(q+1) only if q<Q, V(q-1) only if q>-Q

    # = b*pb + a*pa - c*q^2 + V(q) + pb*(1-pa)*(V(q+1)-V(q)) + pa*(1-pb)*(V(q-1)-V(q)) - pb*pa*(V(q+1)+V(q-1)-2V(q))

    # This doesn't fully separate due to the cross term. But for small p, the cross term is negligible.
    # Alternatively, we can note that for fixed a, the optimal b is:
    # max_b [ b*pb + pb*(1-pa)*V(q+1) + pb*pa*V(q) - pb*(1-pa)*V(q) ]
    # = max_b [ pb * (b + (1-pa)*V(q+1) + pa*V(q) - (1-pa)*V(q)) ]
    # = max_b [ pb * (b + V(q+1) - pa*V(q+1) + pa*V(q) - V(q) + pa*V(q)) ]
    # Hmm this is getting complicated. Let me just do a joint optimization on a reduced grid.

    # Actually, let me reconsider. The key insight is that b and a appear almost independently.
    # For a given q, the optimal b maximizes:
    #   pb * [ b + (1-pa)*ΔV_plus + pa*V(q) - (1-pa)*V(q) ] where ΔV_plus = V(q+1) - V(q)
    # Simplifying: pb * [ b + (1-pa)*V(q+1) + pa*V(q) - (1-pa)*V(q) ]
    # = pb * [ b + V(q+1) - pa*V(q+1) + pa*V(q) - V(q) + pa*V(q) ]
    # = pb * [ b + (V(q+1)-V(q)) - pa*(V(q+1)-V(q)) ]
    # = pb * [ b + (1-pa)*(V(q+1)-V(q)) ]

    # Similarly for a: pa * [ a + (1-pb)*(V(q-1)-V(q)) ]

    # So the joint objective is:
    # pb*(b + (1-pa)*D_plus) + pa*(a + (1-pb)*D_minus) - c*q^2 + V(q)
    # where D_plus = V(q+1)-V(q), D_minus = V(q-1)-V(q)

    # This still has cross terms pb*pa. But we can iterate: fix pa, optimize b, fix pb, optimize a.

    # Given time constraints, let's use a simpler approach: precompute the optimal b and a 
    # for each possible "effective inventory change" and use a lookup.

    # Simpler approach: since the cross terms are small (p is small), approximate by separating:
    # Optimal b: max_b pb * (b + D_plus)  [ignoring the (1-pa) factor]
    # Optimal a: max_a pa * (a + D_minus)

    # This gives a good approximation. Let's go with this.

    V = np.zeros(n_q)  # V[q_index] where q_index = q + Q
    best_bid = np.zeros(n_q)
    best_ask = np.zeros(n_q)

    for n in range(steps - 1, -1, -1):
        V_new = np.zeros(n_q)
        bid_new = np.zeros(n_q)
        ask_new = np.zeros(n_q)

        for qi in range(n_q):
            q = q_levels[qi]
            c = (gamma / 2.0) * sigma * sigma * q * q * dt

            # Compute D_plus and D_minus
            if qi < n_q - 1:  # q < Q
                D_plus = V[qi + 1] - V[qi]
            else:
                D_plus = 0.0
                D_plus_available = False
            if qi > 0:  # q > -Q
                D_minus = V[qi - 1] - V[qi]
            else:
                D_minus = 0.0
                D_minus_available = False

            # Optimize bid depth
            if qi < n_q - 1:
                # max_b pb * (b + D_plus)
                obj_bid = p_fill * (depth_grid + D_plus)
                bi = np.argmax(obj_bid)
                best_b = depth_grid[bi]
                best_pb = p_fill[bi]
            else:
                best_b = 0.0
                best_pb = 0.0

            # Optimize ask depth
            if qi > 0:
                # max_a pa * (a + D_minus)
                obj_ask = p_fill * (depth_grid + D_minus)
                ai = np.argmax(obj_ask)
                best_a = depth_grid[ai]
                best_pa = p_fill[ai]
            else:
                best_a = 0.0
                best_pa = 0.0

            # Compute V_new[qi]
            # V_new = max over b,a of [ b*pb + a*pa - c + (1-pb)(1-pa)V(q) + pb(1-pa)V(q+1) + (1-pb)pa*V(q-1) + pb*pa*V(q) ]
            # Using our chosen b and a:
            if qi < n_q - 1 and qi > 0:
                V_q = V[qi]
                V_q1 = V[qi + 1]
                V_qm1 = V[qi - 1]
                V_new[qi] = best_b * best_pb + best_a * best_pa - c + \
                    (1 - best_pb) * (1 - best_pa) * V_q + \
                    best_pb * (1 - best_pa) * V_q1 + \
                    (1 - best_pb) * best_pa * V_qm1 + \
                    best_pb * best_pa * V_q
            elif qi < n_q - 1:
                # q = Q-1, no ask side (q > -Q is true, wait qi>0 means q>-Q)
                # Actually qi < n_q-1 means q < Q, qi > 0 means q > -Q
                # If qi == n_q-1, q = Q, no bid
                # If qi == 0, q = -Q, no ask
                V_q = V[qi]
                V_q1 = V[qi + 1]
                V_new[qi] = best_b * best_pb - c + \
                    (1 - best_pb) * V_q + \
                    best_pb * V_q1
            elif qi > 0:
                V_q = V[qi]
                V_qm1 = V[qi - 1]
                V_new[qi] = best_a * best_pa - c + \
                    (1 - best_pa) * V_q + \
                    best_pa * V_qm1
            else:
                # q = -Q, no ask, and q < Q so bid is available
                # Wait, if qi == 0, q = -Q, so q > -Q is false, no ask.
                # But q < Q is true (unless Q=0), so bid is available.
                V_q = V[qi]
                V_q1 = V[qi + 1]
                V_new[qi] = best_b * best_pb - c + \
                    (1 - best_pb) * V_q + \
                    best_pb * V_q1

            bid_new[qi] = best_b
            ask_new[qi] = best_a

        V = V_new
        best_bid = bid_new
        best_ask = ask_new

    return best_bid, best_ask


def _get_policy(sigma, A, k, gamma, T, steps, Q):
    """Get or compute the optimal policy for given parameters."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        # Check time budget
        elapsed = time.time() - _START_TIME
        if elapsed > 280:  # Safety margin
            # Return a simple heuristic
            depth = 1.0 / k
            bid = np.full(2 * Q + 1, depth)
            ask = np.full(2 * Q + 1, depth)
            _policy_cache[key] = (bid, ask)
            return _policy_cache[key]
        
        bid, ask = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid, ask)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy lookup."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    bid, ask = _get_policy(sigma, A, k, gamma, T, steps, Q)

    # Map inventory to index: q in [-Q, Q] -> index q + Q
    q_indices = (inventory + Q).astype(np.int64)
    # Clamp to valid range
    q_indices = np.clip(q_indices, 0, 2 * Q)

    bid_depths = bid[q_indices].astype(np.float64)
    ask_depths = ask[q_indices].astype(np.float64)

    # Ensure no negative depths or invalid values
    bid_depths = np.maximum(bid_depths, 1e-6)
    ask_depths = np.maximum(ask_depths, 1e-6)

    # At q = Q, no bid; at q = -Q, no ask
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
