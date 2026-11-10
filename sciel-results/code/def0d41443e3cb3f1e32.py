# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_M_vectorized(domain, values, x_max):
    """Compute M(x) = sum_k f(k) * floor(x/k) for x = 1..x_max using vectorized numpy."""
    M = np.zeros(x_max, dtype=np.float64)
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    for k, f in zip(domain, values):
        if f != 0.0:
            M += f * (x_vals // k)
    return M


def prime_number_theorem_partial_function():
    """
    Maximize a = -sum_k f(k)*log(k)/k subject to M(x) <= 1.0001 for all x in [1, 10K].
    
    Key insight: After the S-correction, g(k) = f(k) for k!=1, g(1) = f(1) - S,
    and sum g(k)/k = 0. Then M(x) = sum_{k!=1} g(k)*(floor(x/k) - x/k).
    Since floor(x/k) - x/k <= 0, making g(k) negative makes M(x) positive.
    We want g(k) as negative as possible (bounded by -10 after clipping) for large k.
    
    Strategy: Set f(k) = -10 for a set of k values (k >= 2), f(1) = 0.
    After S-correction: g(k) = -10 for k in set, g(1) = 10*sum_{k in set} 1/k.
    M(x) = sum_{k in set} (-10)*(floor(x/k) - x/k) = 10*sum_{k in set}(x/k - floor(x/k))
    We need max_x M(x) <= 1.0001.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # The effective M(x) with g(k) = -10 for k in S, g(1) = 10*sum(1/k for k in S):
    # M(x) = sum_{k in S} (-10)*(floor(x/k) - x/k) = 10 * sum_{k in S} (x/k - floor(x/k))
    # = 10 * sum_{k in S} frac(x/k) where frac is the fractional part
    # frac(x/k) = x/k - floor(x/k), which is in [0, 1)
    # So M(x) = 10 * sum_{k in S} frac(x/k)
    # Max possible M(x) = 10 * |S| (if all frac = 1, but that's impossible simultaneously)
    # We need 10 * sum frac(x/k) <= 1.0001, so sum frac(x/k) <= 0.10001
    
    # For a single k: max frac(x/k) over x in [1, 10K] is (k-1)/k (at x = k-1 or similar)
    # Actually frac(x/k) = (x mod k)/k, max is (k-1)/k
    # So for single k, M_max = 10*(k-1)/k. For this <= 1.0001, need 10*(k-1)/k <= 1.0001
    # => (k-1)/k <= 0.10001 => 1 - 1/k <= 0.10001 => 1/k >= 0.89999 => k <= 1.111
    # So no single k >= 2 works with f(k) = -10!
    
    # We need to use smaller magnitude or a combination.
    # Let's use f(k) = -c for k in S, where c < 10.
    # Then M(x) = c * sum_{k in S} frac(x/k), need c * max_x sum frac(x/k) <= 1.0001
    # Objective: a = c * sum_{k in S} log(k)/k
    
    # To maximize, we want large c and large sum log(k)/k.
    # But c is limited by the constraint.
    
    # Alternative: use f(k) = -10 but with a very small set S where the fracs don't align.
    # For |S| = 1: c_max = 1.0001 / ((k-1)/k) = 1.0001*k/(k-1)
    #   For k=2: c_max = 1.0001*2/1 = 2.0002, but clipped to 10, so c=2.0002
    #   a = 2.0002 * log(2)/2 = 2.0002 * 0.3466 = 0.6933
    # For k=3: c_max = 1.0001*3/2 = 1.50015, a = 1.50015*log(3)/3 = 1.50015*0.3662 = 0.5494
    # For k=2, a = 0.6933 is better.
    
    # For |S| = 2, say S = {2, 3}:
    # M(x) = c * (frac(x/2) + frac(x/3))
    # Max of frac(x/2) + frac(x/3) over x: 
    #   x mod 6 determines it. Values of (frac(x/2), frac(x/3)) for x mod 6:
    #   0: (0, 0), 1: (0.5, 0.333), 2: (0, 0.667), 3: (0.5, 0), 4: (0, 0.333), 5: (0.5, 0.667)
    #   Max sum = 0.5 + 0.667 = 1.167 at x mod 6 = 1 or 5
    # c_max = 1.0001 / 1.167 = 0.857
    # a = 0.857 * (log(2)/2 + log(3)/3) = 0.857 * (0.3466 + 0.3662) = 0.857 * 0.7128 = 0.611
    # Less than single k=2.
    
    # For S = {2, 5}:
    # frac(x/2) + frac(x/5), x mod 10:
    # 0:(0,0), 1:(.5,.2), 2:(0,.4), 3:(.5,.6), 4:(0,.8), 5:(.5,0), 6:(0,.2), 7:(.5,.4), 8:(0,.6), 9:(.5,.8)
    # Max = .5+.8 = 1.3 at x mod 10 = 4,8,9
    # c_max = 1.0001/1.3 = 0.769
    # a = 0.769*(log2/2 + log5/5) = 0.769*(0.3466+0.3219) = 0.769*0.6685 = 0.514
    
    # It seems single k=2 with c=2.0002 gives the best so far: a ≈ 0.6933
    
    # But wait, we can also use positive f(k) values! If f(k) > 0, then g(k) > 0,
    # and g(k)*(floor(x/k)-x/k) < 0, which REDUCES M(x). This gives us slack to make
    # other g(k') more negative.
    
    # Better strategy: Use a mix. Set f(k) = -c_neg for some k, and f(j) = +c_pos for others.
    # The positive values reduce M(x), allowing larger negative values elsewhere.
    
    # Actually, the cleanest approach: solve the LP numerically for a moderate K.
    # But LP for K=200000 is too large. Let's use a greedy heuristic.
    
    # Greedy: Start with all f(k) = 0. For each k from K down to 2, try to decrease f(k)
    # (make it more negative) by the maximum amount that keeps M(x) <= 1.0001.
    # Also allow increasing f(k) (making it positive) if that helps.
    
    # But this is O(K * 10K * m) which is too slow for large K.
    
    # Let me try a different approach: use the fact that for the "pure" case where
    # only f(1) and f(2) are non-zero:
    # g(2) = f(2), g(1) = -f(2)/2 (from sum g(k)/k = 0)
    # M(x) = g(1)*x + g(2)*floor(x/2) = -f(2)*x/2 + f(2)*floor(x/2) = f(2)*(floor(x/2) - x/2)
    # = -f(2)*frac(x/2)
    # For M(x) <= 1.0001: if f(2) > 0, M(x) = -f(2)*frac(x/2) <= 0 < 1.0001. Always feasible!
    # If f(2) < 0, M(x) = |f(2)|*frac(x/2), max = |f(2)|*0.5. Need |f(2)|*0.5 <= 1.0001,
    # so |f(2)| <= 2.0002. With f(2) = -2.0002, a = -f(2)*log(2)/2 = 2.0002*0.3466 = 0.6933.
    
    # With f(2) > 0: a = -f(2)*log(2)/2 < 0. Not good.
    
    # So for k=2, best is f(2) = -2.0002, a = 0.6933.
    
    # Now add k=3. Let f(2) = a2, f(3) = a3.
    # g(2) = a2, g(3) = a3, g(1) = -a2/2 - a3/3
    # M(x) = g(1)*x + g(2)*floor(x/2) + g(3)*floor(x/3)
    # = (-a2/2 - a3/3)*x + a2*floor(x/2) + a3*floor(x/3)
    # = a2*(floor(x/2) - x/2) + a3*(floor(x/3) - x/3)
    # = -a2*frac(x/2) - a3*frac(x/3)
    # Need -a2*frac(x/2) - a3*frac(x/3) <= 1.0001 for all x.
    # If a2 < 0 and a3 < 0: a2*frac(x/2) + a3*frac(x/3) >= a2*1 + a3*1 (not quite, fracs < 1)
    # Max of a2*frac(x/2) + a3*frac(x/3) when a2,a3 < 0:
    # = |a2|*frac(x/2) + |a3|*frac(x/3), max over x.
    # For x mod 6: frac(x/2) in {0, 0.5}, frac(x/3) in {0, 1/3, 2/3}
    # Max sum = 0.5 + 2/3 = 7/6 (at x mod 6 = 1, 5)
    # So |a2|*0.5 + |a3|*2/3 <= 1.0001 (approximately, the max is 7/6 of the "unit" sum)
    # Actually max of |a2|*frac(x/2) + |a3|*frac(x/3) = max over x of that expression.
    # At x mod 6 = 1: |a2|*0.5 + |a3|*1/3
    # At x mod 6 = 5: |a2|*0.5 + |a3|*2/3
    # At x mod 6 = 3: |a2|*0.5 + |a3|*0
    # Max is at x mod 6 = 5: 0.5*|a2| + (2/3)*|a3| <= 1.0001
    # Objective: a = |a2|*log(2)/2 + |a3|*log(3)/3
    # Maximize |a2|*0.3466 + |a3|*0.3662 subject to 0.5*|a2| + 0.667*|a3| <= 1.0001
    # |a2| <= 2.0002, |a3| <= 1.5001 (from clipping, but these are < 10 so OK)
    # This is a linear program. The optimum is at a vertex.
    # If we set |a3| = 0: |a2| = 2.0002, a = 0.6933
    # If we set |a2| = 0: |a3| = 1.0001/0.667 = 1.5002, a = 1.5002*0.3662 = 0.5494
    # Mixed: 0.5*|a2| + 0.667*|a3| = 1.0001
    # a = |a2|*0.3466 + |a3|*0.3662
    # |a2| = (1.0001 - 0.667*|a3|)/0.5 = 2.0002 - 1.334*|a3|
    # a = (2.0002 - 1.334*|a3|)*0.3466 + |a3|*0.3662
    # = 0.6933 - 0.4624*|a3| + 0.3662*|a3|
    # = 0.6933 - 0.0962*|a3|
    # This decreases with |a3|! So best is |a3| = 0, i.e., only use k=2.
    
    # This suggests that for this formulation, only using k=2 is optimal among {2,3}.
    
    # Let me try k=2 and k=4:
    # M(x) = -a2*frac(x/2) - a4*frac(x/4) (with a2,a4 < 0)
    # frac(x/2) in {0, 0.5}, frac(x/4) in {0, 0.25, 0.5, 0.75}
    # x mod 4: 0:(0,0), 1:(0.5,0.25), 2:(0,0.5), 3:(0.5,0.75)
    # Max: 0.5+0.75 = 1.25 at x mod 4 = 3
    # Constraint: 0.5*|a2| + 0.75*|a4| <= 1.0001 (approx, need to check all x)
    # Actually max over x of |a2|*frac(x/2) + |a4|*frac(x/4):
    # x mod 4 = 1: 0.5|a2| + 0.25|a4|
    # x mod 4 = 2: 0 + 0.5|a4|
    # x mod 4 = 3: 0.5|a2| + 0.75|a4|
    # Max is 0.5|a2| + 0.75|a4|
    # Objective: |a2|*log2/2 + |a4|*log4/4 = |a2|*0.3466 + |a4|*0.1733
    # |a2| = (1.0001 - 0.75|a4|)/0.5 = 2.0002 - 1.5|a4|
    # a = (2.0002-1.5|a4|)*0.3466 + |a4|*0.1733 = 0.6933 - 0.5199|a4| + 0.1733|a4| = 0.6933 - 0.3466|a4|
    # Again decreases! Best is |a4|=0.
    
    # Pattern: adding more k's with the same sign doesn't help because log(k)/k decreases.
    
    # What if we use POSITIVE values for some k? 
    # If a3 > 0, then -a3*frac(x/3) < 0, which reduces M(x).
    # This gives slack to make a2 more negative.
    # But a = -a2*log2/2 - a3*log3/3. With a3 > 0, the a3 term is negative, hurting a.
    # But it allows a2 to be more negative, potentially increasing a overall.
    
    # Let's try: a2 < 0, a3 > 0.
    # M(x) = |a2|*frac(x/2) - a3*frac(x/3) (since a3 > 0, -a3*frac < 0)
    # Max over x: need max_x (|a2|*frac(x/2) - a3*frac(x/3)) <= 1.0001
    # x mod 6: 
    # 0: 0, 1: 0.5|a2| - a3/3, 2: 0 - 2a3/3, 3: 0.5|a2| - 0, 4: 0 - a3/3, 5: 0.5|a2| - 2a3/3
    # Max is 0.5|a2| (at x mod 6 = 3)
    # So constraint: 0.5|a2| <= 1.0001, |a2| <= 2.0002. Same as before!
    # a = |a2|*0.3466 - a3*0.3662. To maximize, set a3 = 0.
    
    # What about a2 < 0, a3 < 0, but also a5 > 0?
    # This is getting complex. Let me just try the simple case: f(2) = -2.0002, rest 0.
    # a = 0.6933.
    
    # Can we do better with a different structure?
    # What if we use many k's with small negative values?
    # If f(k) = -epsilon for k = 2..K, then:
    # M(x) = epsilon * sum_{k=2}^K frac(x/k)
    # Max over x of sum_{k=2}^K frac(x/k). For large K, this is approximately K/2 (average frac is 1/2).
    # So M_max ≈ epsilon * K/2. Need epsilon*K/2 <= 1.0001, so epsilon <= 2.0002/K.
    # a = epsilon * sum_{k=2}^K log(k)/k ≈ epsilon * (1/2)(log K)^2 (by integral approximation)
    # = (2.0002/K) * (1/2)(log K)^2 = 1.0001 * (log K)^2 / K
    # This is maximized at K = e^2 ≈ 7.39, giving a ≈ 1.0001 * 4/7.39 ≈ 0.541.
    # Less than 0.6933 from single k=2.
    
    # So the single k=2 solution with a ≈ 0.6933 seems strong.
    # But can we do even better?
    
    # What about f(2) = -10 (the clip limit)? Then M(x) = 10*frac(x/2), max = 5.
    # 5 > 1.0001, infeasible.
    
    # The constraint is the binding one. With only k=2, max feasible |f(2)| = 2.0002.
    
    # Let me try to use the full LP for a small set of k's to see if there's a better solution.
    # For k in {2, 3, 4, 5, 6, 7, 8, 9, 10}:
    # Variables: f(k) for k=2..10, with f(1) determined by sum constraint.
    # M(x) = sum_{k=2}^{10} f(k)*(floor(x/k) - x/k) for x in [1, 100]
    # Maximize -sum f(k)*log(k)/k subject to M(x) <= 1.0001 and |f(k)| <= 10.
    
    # This is a linear program with 9 variables and 100+ constraints.
    # Let me solve it with scipy.
    
    from scipy.optimize import linprog
    
    # Variables: f2, f3, ..., f10 (9 variables)
    # Objective: minimize sum f(k)*log(k)/k (since we want to maximize -sum)
    # Constraints: for each x in 1..100: sum f(k)*(floor(x/k) - x/k) <= 1.0001
    #              -10 <= f(k) <= 10
    
    K_small = 10
    ks = np.arange(2, K_small + 1)
    x_max_small = 10 * K_small  # 100
    
    # Coefficient matrix for M(x) constraints
    # A_eq part: for each x, row is [floor(x/k) - x/k for k in ks]
    A_ub = np.zeros((x_max_small, len(ks)))
    for i, x in enumerate(range(1, x_max_small + 1)):
        for j, k in enumerate(ks):
            A_ub[i, j] = np.floor(x / k) - x / k
    
    b_ub = np.full(x_max_small, 1.0001)
    
    # Objective: c_j = log(k_j) / k_j (we minimize sum c_j * f_j = sum f_j log(k_j)/k_j)
    # Wait, we want to maximize -sum f(k)*log(k)/k, so minimize sum f(k)*log(k)/k
    c_obj = np.log(ks.astype(np.float64)) / ks.astype(np.float64)
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10, 10)] * len(ks)
    
    # Solve LP
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        f_vals = result.x
        # Construct full domain and values
        domain = np.concatenate([[1], ks])
        # f(1) is determined by sum f(k)/k = 0 => f(1) = -sum_{k>=2} f(k)/k
        f1 = -np.sum(f_vals / ks.astype(np.float64))
        values = np.concatenate([[f1], f_vals])
        
        # Clip to [-10, 10]
        values = np.clip(values, -10, 10)
        
        # Verify feasibility
        K_max = int(domain[-1])
        x_max_eval = 10 * K_max
        M = _compute_M_vectorized(domain, values, x_max_eval)
        if np.max(M) <= 1.0001:
            # Compute objective
            log_k = np.log(domain.astype(np.float64))
            log_k[0] = 0.0
            a_val = -np.sum(values * log_k / domain.astype(np.float64))
            print(f"LP solution: a = {a_val}, f_vals = {f_vals}")
            return domain, values
    
    # Fallback: single k=2 solution
    domain = np.array([1, 2], dtype=np.int64)
    f2 = -2.0002
    f1 = -f2 / 2  # = 1.0001
    values = np.array([f1, f2], dtype=np.float64)
    
    # Verify
    x_max_eval = 20
    M = _compute_M_vectorized(domain, values, x_max_eval)
    print(f"Fallback: max M = {np.max(M)}, a = {-np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))}")
    
    return domain, values


# EVOLVE-BLOCK-END
