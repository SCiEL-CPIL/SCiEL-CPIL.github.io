"""
Strategy: Analytic Solution to the Bellman Equation.
Instead of grid search, we derive the optimal bid and ask depths analytically.
The expected reward for a given state (t, q) and actions (b, a) is:
J(b, a) = (1-pb)(1-pa)V_{t+1}(q) + pb(1-pa)(b + V_{t+1}(q+1)) + (1-pb)pa(a + V_{t+1}(q-1)) + pb pa (b + a + V_{t+1}(q)) - C

Where pb = 1 - exp(-lambda_b), lambda_b = A exp(-k b) dt.
Let D_b = V_{t+1}(q+1) - V_{t+1}(q) and D_a = V_{t+1}(q) - V_{t+1}(q-1).
The term involving b simplifies to:
pb * [ (1-pa)(b + D_b) + pa(b) ] = pb * [ b + (1-pa)D_b ]
Similarly for a:
pa * [ (1-pb)a + pb(a + D_a) ] = pa * [ a + (1-pb)D_a ]

So J = V_q + pb(1-pa)(b + D_b) + (1-pb)pa(a + D_a) + pb pa (b + a) - C
Wait, let's re-expand carefully.
J = (1-pb)(1-pa)V_q + pb(1-pa)(b + V_{q+1}) + (1-pb)pa(a + V_{q-1}) + pb pa (b + a + V_q) - C
J = V_q [ (1-pb)(1-pa) + pb pa ] + pb(1-pa)b + pb(1-pa)V_{q+1} + (1-pb)pa a + (1-pb)pa V_{q-1} + pb pa b + pb pa a - C
J = V_q [ 1 - pb - pa + 2 pb pa ] + pb b [ (1-pa) + pa ] + pa a [ (1-pb) + pb ] + pb(1-pa)V_{q+1} + (1-pb)pa V_{q-1} - C
J = V_q [ 1 - pb - pa + 2 pb pa ] + pb b + pa a + pb(1-pa)V_{q+1} + (1-pb)pa V_{q-1} - C

Let's group by pb and pa.
J = V_q - V_q pb - V_q pa + 2 V_q pb pa + pb b + pa a + pb V_{q+1} - pb pa V_{q+1} + pa V_{q-1} - pb pa V_{q-1} - C
J = V_q - C + pb [ b - V_q + V_{q+1} - pa(V_{q+1} - V_{q-1})? No.
Let's keep it as a function of pb and pa.
J(pb, pa) = V_q - C + pb [ b + V_{q+1} - V_q - pa(V_{q+1} + V_{q-1} - 2V_q)? No.

Let's use the derivative directly on the simplified form:
J = V_q + pb(1-pa)(b + V_{q+1}) + (1-pb)pa(a + V_{q-1}) + pb pa (b + a + V_q) - C
J = V_q + pb(1-pa)b + pb(1-pa)V_{q+1} + (1-pb)pa a + (1-pb)pa V_{q-1} + pb pa b + pb pa a + pb pa V_q - C
J = V_q + pb b + pb pa a + pb V_{q+1} - pb pa V_{q+1} + pa a - pb pa a + pa V_{q-1} - pb pa V_{q-1} + pb pa V_q - C
J = V_q - C + pb b + pa a + pb V_{q+1} + pa V_{q-1} - pb pa (V_{q+1} + V_{q-1} - V_q)

Let Delta = V_{q+1} + V_{q-1} - V_q.
J(pb, pa) = (V_q - C) + pb (b + V_{q+1}) + pa (a + V_{q-1}) - pb pa Delta

To maximize J w.r.t b (and thus pb):
dJ/db = (dJ/dpb) * (dpb/db)
dpb/db = -k * A * exp(-k b) * dt * exp(-k b)? No.
pb = 1 - exp(-A exp(-k b) dt)
dpb/db = - exp(-A exp(-k b) dt) * (-A dt) * d(exp(-k b))/db
dpb/db = exp(-A exp(-k b) dt) * A dt * (-k) exp(-k b)
dpb/db = -k pb' ? No.
Let L_b = A exp(-k b) dt.
pb = 1 - e^{-L_b}.
dpb/db = -e^{-L_b} * (-dL_b/db) = e^{-L_b} * dL_b/db.
dL_b/db = L_b * (-k).
So dpb/db = -k L_b e^{-L_b} = -k (1-pb) L_b? No.
e^{-L_b} = 1 - pb.
So dpb/db = -k L_b (1 - pb).
Also L_b = -ln(1-pb)/dt * 1/A? No.
L_b = -ln(1-pb) / (A dt)? No.
1 - pb = exp(-A exp(-k b) dt).
ln(1-pb) = -A exp(-k b) dt.
exp(-k b) = -ln(1-pb) / (A dt).
L_b = A dt exp(-k b) = -ln(1-pb).
So dpb/db = -k (-ln(1-pb)) (1-pb) = k ln(1-pb) (1-pb).
Wait, ln(1-pb) is negative. So dpb/db is negative. Correct.

Now, dJ/dpb = b + V_{q+1} - pa Delta.
Set dJ/db = 0 => dJ/dpb * dpb/db = 0.
Since dpb/db != 0, we need dJ/dpb = 0.
b + V_{q+1} - pa Delta = 0 => b = pa Delta - V_{q+1}.
Similarly for a:
dJ/dpa = a + V_{q-1} - pb Delta = 0 => a = pb Delta - V_{q-1}.

This gives a system of two equations:
b = pa Delta - V_{q+1}
a = pb Delta - V_{q-1}

Where pb = 1 - exp(-k A dt exp(-k b)) and pa = 1 - exp(-k A dt exp(-k a)).
This is a coupled nonlinear system. We can solve it using fixed-point iteration or Newton's method.
Given the small state space, we can solve this for each (t, q) state.
Fixed point iteration:
Initialize b=0, a=0.
Loop:
  pb = 1 - exp(-A dt exp(-k b))
  pa = 1 - exp(-A dt exp(-k a))
  b_new = pa Delta - V_{q+1}
  a_new = pb Delta - V_{q-1}
  Clip b_new, a_new to [0, max_depth].
  Check convergence.

This is much faster than grid search and yields the exact optimum (local max, which is global due to concavity of J in pb, pa? J is bilinear in pb, pa with negative cross term -Delta. If Delta > 0, J is concave in (pb, pa) if -Delta < 0? Hessian [[0, -Delta], [-Delta, 0]] is indefinite. So it's a saddle point?
Wait, J = C0 + pb B + pa A - pb pa Delta.
If Delta > 0, the cross term is negative. This is not jointly concave.
However, the mapping from b to pb is monotonic.
Let's check the nature of the optimum.
If we fix pa, J is linear in pb. Max is at boundary pb=0 or pb=1.
If we fix pb, J is linear in pa. Max is at boundary pa=0 or pa=1.
This suggests the optimum is at the corners of the (pb, pa) unit square?
But pb and pa are continuous variables determined by b and a.
If the function is linear in pb, the maximum over pb in [0,1] is at 0 or 1.
This implies the optimal strategy is to either quote with probability 0 (infinite depth) or probability 1 (zero depth)?
No, pb is determined by b. As b varies, pb varies continuously from 1 (b=0) to 0 (b=infinity).
If J is linear in pb, the max is at the boundaries of the range of pb.
Range of pb is [0, 1].
So optimal pb is 0 or 1?
This would imply b is either 0 or infinity.
This seems counter-intuitive for market making. Let's re-evaluate the linearity.
J(pb, pa) = V_q - C + pb (b + V_{q+1}) + pa (a + V_{q-1}) - pb pa Delta.
Note that b depends on pb! b = -ln(1-pb)/(k A dt).
So J is NOT linear in pb. The term pb * b becomes pb * (-ln(1-pb)/(k A dt)).
Let f(pb) = pb (-ln(1-pb)).
f'(pb) = -ln(1-pb) + pb/(1-pb).
f''(pb) = 1/(1-pb) - (1-pb + pb)/(1-pb)^2 = 1/(1-pb) - 1/(1-pb)^2 = (1-pb-1)/(1-pb)^2 = -pb/(1-pb)^2 < 0.
So the term pb*b is concave in pb.
The term -pb pa Delta is bilinear.
If Delta > 0, the Hessian of the cross term is [[0, -Delta], [-Delta, 0]].
Total Hessian:
H = [[f''(pb), -Delta], [-Delta, f''(pa)]]
For concavity, we need H negative semi-definite.
f''(pb) < 0.
Det(H) = f''(pb)f''(pa) - Delta^2.
If Delta is large, Det can be negative, meaning indefinite.
However, in practice, Delta (V_{q+1} + V_{q-1} - V_q) is likely small compared to the curvature of the log term?
Or maybe the optimum is still interior.
We can just use the fixed-point iteration on the first-order conditions derived earlier, but accounting for the dependency of b on pb?
No, the FOC was derived treating b and pb as linked.
dJ/db = 0 led to b = pa Delta - V_{q+1}.
This equation does not involve the derivative of b w.r.t pb explicitly because we used the chain rule correctly?
Let's re-verify dJ/db.
J(b, a) = V_q - C + pb(b) b + pa(a) a + pb(b) V_{q+1} + pa(a) V_{q-1} - pb(b) pa(a) Delta.
dJ/db = (dpb/db) b + pb + (dpb/db) V_{q+1} - (dpb/db) pa Delta.
Set to 0:
dpb/db (b + V_{q+1} - pa Delta) + pb = 0.
dpb/db = -k L_b (1-pb).
So -k L_b (1-pb) (b + V_{q+1} - pa Delta) + pb = 0.
This is the correct FOC.
Similarly for a:
-k L_a (1-pa) (a + V_{q-1} - pb Delta) + pa = 0.

We can solve this system using Newton's method or fixed point.
Fixed point:
b = pa Delta - V_{q+1} - pb / (dpb/db)
a = pb Delta - V_{q-1} - pa / (dpa/da)

Since dpb/db is negative, -pb/(dpb/db) is positive.
Let's implement this iterative solver.
"""

import numpy as np
import time

_policy_cache = {}

def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    V_next = np.zeros(n_states)
    
    # Constants for the solver
    max_iter = 50
    tol = 1e-8
    
    for t in range(steps - 1, -1, -1):
        V_current = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_values[q_idx]
            
            allow_bid = (q < Q)
            allow_ask = (q > -Q)
            
            v_q = V_next[q_idx]
            
            if allow_bid:
                v_qp = V_next[q_idx + 1]
            else:
                v_qp = 0.0
                # If not allowed, we effectively set pb=0, so b is irrelevant.
                # We will force b to a large value (or 0 with pb=0 logic)
                
            if allow_ask:
                v_qm = V_next[q_idx - 1]
            else:
                v_qm = 0.0

            Delta = v_qp + v_qm - v_q
            
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # If neither allowed, V_current = v_q - cost
            if not allow_bid and not allow_ask:
                V_current[q_idx] = v_q - cost
                opt_bid[t, q_idx] = 0.0
                opt_ask[t, q_idx] = 0.0
                continue
            
            # Initialize b, a
            b = 0.0
            a = 0.0
            
            # Iterative solver
            for _ in range(max_iter):
                # Calculate pb, pa
                if allow_bid:
                    pb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    dpb_db = -k * A * np.exp(-k * b) * dt * np.exp(-A * np.exp(-k * b) * dt)
                    # dpb_db = -k * L_b * (1-pb)
                    # L_b = A exp(-k b) dt
                    # 1-pb = exp(-L_b)
                    # dpb_db = -k L_b (1-pb)
                else:
                    pb = 0.0
                    dpb_db = 0.0 # Unused
                    
                if allow_ask:
                    pa = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    dpa_da = -k * A * np.exp(-k * a) * dt * np.exp(-A * np.exp(-k * a) * dt)
                else:
                    pa = 0.0
                    dpa_da = 0.0

                # Compute new b, a from FOCs
                # FOC for b: dpb_db (b + v_qp - pa Delta) + pb = 0
                # b + v_qp - pa Delta = -pb / dpb_db
                # b = -pb / dpb_db - v_qp + pa Delta
                
                b_new = b
                a_new = a
                
                if allow_bid:
                    if abs(dpb_db) > 1e-12:
                        b_new = -pb / dpb_db - v_qp + pa * Delta
                    else:
                        b_new = b
                        # If dpb_db is 0, it's at a boundary or flat.
                        # Usually dpb_db is 0 when b->inf (pb->0) or b->0 (pb->1, dpb_db-> -k A dt).
                        # At b=0, dpb_db is max magnitude.
                        # If dpb_db is small, we are far out.
                    
                    # Clip b_new to valid range [0, 2.0]
                    if b_new < 0.0:
                        b_new = 0.0
                    elif b_new > 2.0:
                        b_new = 2.0
                        
                if allow_ask:
                    if abs(dpa_da) > 1e-12:
                        a_new = -pa / dpa_da - v_qm + pb * Delta
                    else:
                        a_new = a
                    if a_new < 0.0:
                        a_new = 0.0
                    elif a_new > 2.0:
                        a_new = 2.0

                if abs(b_new - b) < tol and abs(a_new - a) < tol:
                    b = b_new
                    a = a_new
                    break
                
                b = b_new
                a = a_new
            
            opt_bid[t, q_idx] = b
            opt_ask[t, q_idx] = a
            
            # Calculate V_current
            # J = v_q - cost + pb b + pa a + pb v_qp + pa v_qm - pb pa Delta
            # Note: if not allow_bid, pb=0, b=0 (or whatever, term vanishes)
            # if not allow_ask, pa=0, a=0
            
            J = v_q - cost
            if allow_bid:
                J += pb * (b + v_qp)
            if allow_ask:
                J += pa * (a + v_qm)
            if allow_bid and allow_ask:
                J -= pb * pa * Delta
                
            V_current[q_idx] = J
            
        V_next = V_current
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
