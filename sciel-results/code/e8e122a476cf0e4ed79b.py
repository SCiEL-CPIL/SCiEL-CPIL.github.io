"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, brentq

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _two_state_dp(p01, p10, lam):
    """
    Dynamic programming for two-state Markov chain.
    States: 0 (low value A), 1 (high value B)
    p01: prob of transitioning from state 0 to state 1
    p10: prob of transitioning from state 1 to state 0
    lam: Lagrange multiplier for sum constraint
    
    Returns: (h_array, sum_h, max_overlap)
    """
    A = 0.381966  # (3-sqrt(5))/2
    B = 1.0 - A   # ~0.618034
    
    # Reward for being in state s at position i:
    # -lam * value(s) to encourage correct sum
    # Plus penalty for overlap with previous state
    # We want to minimize max overlap, so use a quadratic penalty
    
    # DP: dp[i][s] = min cost to reach state s at position i
    # Cost includes overlap penalty + Lagrange term
    
    INF = 1e10
    dp = np.zeros((N, 2))
    parent = np.zeros((N, 2), dtype=int)
    
    # Initial state
    dp[0, 0] = -lam * A
    dp[0, 1] = -lam * B
    
    for i in range(1, N):
        for s in range(2):
            val_s = A if s == 0 else B
            # Overlap with previous position: if prev state was s_prev, 
            # overlap at position i is val_s * (1 - val_{s_prev})
            best_cost = INF
            best_prev = 0
            for s_prev in range(2):
                val_prev = A if s_prev == 0 else B
                # Transition probability penalty: -log(p) to encourage transitions
                if s_prev == 0 and s == 1:
                    trans_pen = -math.log(max(p01, 1e-15))
                elif s_prev == 1 and s == 0:
                    trans_pen = -math.log(max(p10, 1e-15))
                else:
                    trans_pen = 0.0
                
                overlap = val_s * (1.0 - val_prev)
                # Use quadratic penalty for overlap to make it smooth
                cost = dp[i-1, s_prev] + trans_pen + 100.0 * overlap * overlap - lam * val_s
                if cost < best_cost:
                    best_cost = cost
                    best_prev = s_prev
            dp[i, s] = best_cost
            parent[i, s] = best_prev
    
    # Find best final state
    best_final = np.argmin(dp[N-1])
    
    # Backtrack
    states = np.zeros(N, dtype=int)
    states[N-1] = best_final
    for i in range(N-1, 0, -1):
        states[i-1] = parent[i, states[i]]
    
    # Convert to h values
    h = np.where(states == 0, A, B)
    
    return h, np.sum(h), _score(h)

def _fix_sum(h):
    """Fix the sum of h to be exactly N/2."""
    for _ in range(10):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _markov_objective(params):
    """
    Objective for Markov chain optimization.
    params = [p01, p10, lam]
    """
    p01, p10, lam = params
    p01 = np.clip(p01, 1e-6, 1.0 - 1e-6)
    p10 = np.clip(p10, 1e-6, 1.0 - 1e-6)
    
    h, sum_h, max_ov = _two_state_dp(p01, p10, lam)
    
    # Penalize sum constraint violation
    sum_penalty = 100.0 * (sum_h - N/2.0)**2
    
    return max_ov + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()
    
    # Phase 1: Optimize Markov chain parameters
    try:
        if time.time() < deadline - 20.0:
            # Initial guess for Markov parameters
            # We want roughly equal time in each state
            # p01 and p10 determine the stationary distribution
            # Stationary: pi_0 = p10/(p01+p10), pi_1 = p01/(p01+p10)
            # For equal time: p01 = p10
            
            # Try different initial points
            initial_points = [
                [0.1, 0.1, 0.0],    # Low transition rates
                [0.3, 0.3, 0.0],    # Medium transition rates
                [0.5, 0.5, 0.0],    # High transition rates
                [0.2, 0.4, 0.0],    # Asymmetric
                [0.4, 0.2, 0.0],    # Asymmetric opposite
            ]
            
            best_markov_score = current_score
            best_markov_params = None
            
            for ip in initial_points:
                if time.time() > deadline - 15.0:
                    break
                
                try:
                    result = minimize(
                        _markov_objective,
                        np.array(ip),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                    )
                    
                    if result.fun < best_markov_score:
                        p01, p10, lam = result.x
                        p01 = np.clip(p01, 1e-6, 1.0 - 1e-6)
                        p10 = np.clip(p10, 1e-6, 1.0 - 1e-6)
                        
                        h_markov, sum_h, max_ov = _two_state_dp(p01, p10, lam)
                        h_markov = _fix_sum(h_markov)
                        new_score = _score(h_markov)
                        
                        if new_score < best_markov_score:
                            best_markov_score = new_score
                            best_markov_params = (p01, p10, lam)
                            h_final = h_markov.copy()
                            current_score = new_score
                            
                except Exception:
                    pass
            
    except Exception:
        pass
    
    # Phase 2: Fine-tune using SLSQP on the full h vector, starting from Markov solution
    try:
        if time.time() < deadline - 10.0:
            def objective_raw(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_raw,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )

            if result.fun < current_score:
                h_opt = result.x.copy()
                h_opt = _fix_sum(h_opt)
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt.copy()
                    current_score = new_score

    except Exception:
        pass
    
    # Phase 3: Try different A, B values with Markov chain
    try:
        if time.time() < deadline - 5.0:
            # Sweep over possible A values
            for A_test in [0.3, 0.35, 0.381966, 0.4, 0.45, 0.5]:
                if time.time() > deadline - 3.0:
                    break
                B_test = 1.0 - A_test
                
                # Optimize Markov parameters for this A, B
                def markov_obj_ab(params):
                    p01, p10, lam = params
                    p01 = np.clip(p01, 1e-6, 1.0 - 1e-6)
                    p10 = np.clip(p10, 1e-6, 1.0 - 1e-6)
                    
                    # DP with specific A, B
                    INF = 1e10
                    dp = np.zeros((N, 2))
                    parent = np.zeros((N, 2), dtype=int)
                    
                    dp[0, 0] = -lam * A_test
                    dp[0, 1] = -lam * B_test
                    
                    for i in range(1, N):
                        for s in range(2):
                            val_s = A_test if s == 0 else B_test
                            best_cost = INF
                            best_prev = 0
                            for s_prev in range(2):
                                val_prev = A_test if s_prev == 0 else B_test
                                if s_prev == 0 and s == 1:
                                    trans_pen = -math.log(max(p01, 1e-15))
                                elif s_prev == 1 and s == 0:
                                    trans_pen = -math.log(max(p10, 1e-15))
                                else:
                                    trans_pen = 0.0
                                
                                overlap = val_s * (1.0 - val_prev)
                                cost = dp[i-1, s_prev] + trans_pen + 100.0 * overlap * overlap - lam * val_s
                                if cost < best_cost:
                                    best_cost = cost
                                    best_prev = s_prev
                            dp[i, s] = best_cost
                            parent[i, s] = best_prev
                    
                    best_final = np.argmin(dp[N-1])
                    states = np.zeros(N, dtype=int)
                    states[N-1] = best_final
                    for i in range(N-1, 0, -1):
                        states[i-1] = parent[i, states[i]]
                    
                    h = np.where(states == 0, A_test, B_test)
                    sum_h = np.sum(h)
                    max_ov = _score(h)
                    sum_penalty = 100.0 * (sum_h - N/2.0)**2
                    return max_ov + sum_penalty
                
                for ip in [[0.2, 0.2, 0.0], [0.4, 0.4, 0.0]]:
                    if time.time() > deadline - 2.0:
                        break
                    try:
                        result = minimize(
                            markov_obj_ab,
                            np.array(ip),
                            method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                        )
                        
                        if result.fun < current_score:
                            p01, p10, lam = result.x
                            p01 = np.clip(p01, 1e-6, 1.0 - 1e-6)
                            p10 = np.clip(p10, 1e-6, 1.0 - 1e-6)
                            
                            # Reconstruct h
                            INF = 1e10
                            dp = np.zeros((N, 2))
                            parent = np.zeros((N, 2), dtype=int)
                            dp[0, 0] = -lam * A_test
                            dp[0, 1] = -lam * B_test
                            
                            for i in range(1, N):
                                for s in range(2):
                                    val_s = A_test if s == 0 else B_test
                                    best_cost = INF
                                    best_prev = 0
                                    for s_prev in range(2):
                                        val_prev = A_test if s_prev == 0 else B_test
                                        if s_prev == 0 and s == 1:
                                            trans_pen = -math.log(max(p01, 1e-15))
                                        elif s_prev == 1 and s == 0:
                                            trans_pen = -math.log(max(p10, 1e-15))
                                        else:
                                            trans_pen = 0.0
                                        
                                        overlap = val_s * (1.0 - val_prev)
                                        cost = dp[i-1, s_prev] + trans_pen + 100.0 * overlap * overlap - lam * val_s
                                        if cost < best_cost:
                                            best_cost = cost
                                            best_prev = s_prev
                                    dp[i, s] = best_cost
                                    parent[i, s] = best_prev
                            
                            best_final = np.argmin(dp[N-1])
                            states = np.zeros(N, dtype=int)
                            states[N-1] = best_final
                            for i in range(N-1, 0, -1):
                                states[i-1] = parent[i, states[i]]
                            
                            h_markov = np.where(states == 0, A_test, B_test)
                            h_markov = _fix_sum(h_markov)
                            new_score = _score(h_markov)
                            
                            if new_score < current_score:
                                h_final = h_markov.copy()
                                current_score = new_score
                                
                    except Exception:
                        pass
                        
    except Exception:
        pass
    
    return h_final
