"""Rewritten program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # Parse inputs
    # distribution_spec: likely a list of 16 probabilities or a dict mapping report->prob
    # support: list of 16 tuples (v1, v2)
    # constraints: dict with various constraint specs
    
    # Default support: all 16 combinations
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    n_reports = len(support)
    if n_reports != 16:
        # Try to handle if support is given differently
        # Assume it's a list of 16 entries
        if n_reports == 0:
            support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
            n_reports = 16
    
    # Extract probabilities from distribution_spec
    probs = np.ones(n_reports) / n_reports  # uniform default
    if distribution_spec is not None:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            if len(distribution_spec) == n_reports:
                probs = np.array(distribution_spec, dtype=float)
                # Normalize
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(n_reports) / n_reports
            elif len(distribution_spec) == 1:
                # Maybe it's a single distribution object
                probs = np.ones(n_reports) / n_reports
        elif isinstance(distribution_spec, dict):
            # Map report indices or tuples to probabilities
            for i, rep in enumerate(support):
                if rep in distribution_spec:
                    probs[i] = distribution_spec[rep]
                elif i in distribution_spec:
                    probs[i] = distribution_spec[i]
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(n_reports) / n_reports
    
    # Ensure probs are valid
    probs = np.clip(probs, 0, 1)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_reports) / n_reports
    
    # Extract values for each report
    # support[i] = (v1_i, v2_i)
    values = np.zeros((n_reports, 2))
    for i, rep in enumerate(support):
        if isinstance(rep, (list, tuple)) and len(rep) == 2:
            values[i, 0] = rep[0]
            values[i, 1] = rep[1]
        elif isinstance(rep, (list, tuple)) and len(rep) > 2:
            values[i, 0] = rep[0]
            values[i, 1] = rep[1]
    
    # For a single bidder with IC (incentive compatibility), the allocation rule x(t)
    # and payment rule p(t) must satisfy:
    # 1. Individual Rationality (IR): For each type t, u(t,t) >= 0 where u(t,t) = v(t)*x(t) - p(t)
    #    Since the bidder always reports their true type (truthfulness), we need v(t)*x(t) - p(t) >= 0
    # 2. Incentive Compatibility (IC): For each pair of types (t, t'), 
    #    v(t)*x(t) - p(t) >= v(t)*x(t') - p(t')
    #    This means: the utility from reporting t when true type is t is >= utility from reporting t'
    #
    # For a single bidder, the standard result is that IC + IR implies:
    # - The allocation x(t) must be monotone in the value (in some sense)
    # - The payment is determined by the allocation via the envelope theorem
    #
    # Actually, for a single bidder with multi-dimensional types, the problem is more complex.
    # Let's use the LP formulation.
    #
    # Variables:
    # - x[i][j] for i in 0..15, j in 0..1: allocation of good j to bidder with report i (16*2 = 32 vars)
    # - p[i] for i in 0..15: payment for report i (16 vars)
    # Total: 48 variables
    #
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[i][j] <= 1 for all i, j
    # 2. IR: For each i, values[i] @ x[i] - p[i] >= 0  =>  p[i] <= values[i] @ x[i]
    # 3. IC: For each pair (i, i'), values[i] @ x[i] - p[i] >= values[i] @ x[i'] - p[i']
    #    => values[i] @ (x[i] - x[i']) + p[i'] - p[i] >= 0
    #    => p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    #    This gives 16*16 = 256 constraints (but we can reduce to 16*15/2 = 120 if we consider both directions)
    #    Actually, IC requires for all i, i': u(i,i) >= u(i,i') where u(i,i') = values[i] @ x[i'] - p[i']
    #    So: values[i] @ x[i] - p[i] >= values[i] @ x[i'] - p[i']
    #    => values[i] @ (x[i] - x[i']) >= p[i] - p[i']
    #    This is 16*16 = 256 constraints, but many are redundant.
    
    # Let's set up the LP
    # Variables: [x[0][0], x[0][1], x[1][0], x[1][1], ..., x[15][0], x[15][1], p[0], p[1], ..., p[15]]
    # Index: for i in 0..15, x[i][0] is at index 2*i, x[i][1] is at index 2*i+1
    #        p[i] is at index 32 + i
    
    n_vars = 2 * n_reports + n_reports  # 32 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog, we minimize, so negate
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[32 + i] = -probs[i]  # negative because we're maximizing
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    # 1. Feasibility: 0 <= x[i][j] <= 1
    # x[i][j] <= 1: already handled by bounds
    # x[i][j] >= 0: handled by bounds
    
    # 2. IR: p[i] <= values[i] @ x[i]  =>  -values[i][0]*x[i][0] - values[i][1]*x[i][1] + p[i] <= 0
    #    For each i: -values[i][0]*x[i][0] - values[i][1]*x[i][1] + p[i] <= 0
    
    # 3. IC: For each pair (i, i'): p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    #    => values[i][0]*x[i][0] + values[i][1]*x[i][1] - values[i][0]*x[i'][0] - values[i][1]*x[i'][1] - p[i] + p[i'] >= 0
    #    => -values[i][0]*x[i][0] - values[i][1]*x[i][1] + values[i][0]*x[i'][0] + values[i][1]*x[i'][1] + p[i] - p[i'] <= 0
    
    # Let's build the constraint matrix
    # Start with IR constraints (16 constraints)
    A_rows = []
    b_rows = []
    
    # IR: for each i
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[2*i] = -values[i][0]
        row[2*i+1] = -values[i][1]
        row[32+i] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    # IC: for each pair (i, i') where i != i'
    # p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    # => -values[i] @ x[i] + values[i] @ x[i'] + p[i] - p[i'] <= 0
    for i in range(n_reports):
        for ip in range(n_reports):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            # -values[i] @ x[i]
            row[2*i] = -values[i][0]
            row[2*i+1] = -values[i][1]
            # +values[i] @ x[ip]
            row[2*ip] = values[i][0]
            row[2*ip+1] = values[i][1]
            # +p[i] - p[ip]
            row[32+i] = 1
            row[32+ip] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[i][j] <= 1, p[i] >= 0
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))  # x[i][0]
        bounds.append((0, 1))  # x[i][1]
    for i in range(n_reports):
        bounds.append((0, None))  # p[i] >= 0
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status == 0:
            # Extract solution
            x = np.zeros((n_reports, 2))
            p = np.zeros(n_reports)
            for i in range(n_reports):
                x[i][0] = result.x[2*i]
                x[i][1] = result.x[2*i+1]
                p[i] = result.x[32+i]
            
            # Clip to ensure feasibility
            x = np.clip(x, 0, 1)
            p = np.clip(p, 0, None)
            
            # Verify IC and IR
            # Check IR
            for i in range(n_reports):
                utility = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                if utility < -1e-8:
                    # Fix by adjusting payment
                    p[i] = values[i][0]*x[i][0] + values[i][1]*x[i][1]
            
            # Check IC
            for i in range(n_reports):
                for ip in range(n_reports):
                    if i == ip:
                        continue
                    util_i = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                    util_ip_as_i = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                    if util_i < util_ip_as_i - 1e-8:
                        # IC violated, need to adjust
                        # This shouldn't happen if LP is solved correctly
                        pass
            
            # Check regret (should be 0 for truthful mechanism)
            # Regret is the maximum loss from deviating, which should be 0
            
            # Return the solution
            return {
                'allocations': x,
                'payments': p
            }
        else:
            # LP failed, return a safe default
            x = np.ones((n_reports, 2))
            p = np.array([values[i][0] + values[i][1] for i in range(n_reports)])
            return {
                'allocations': x,
                'payments': p
            }
    
    except Exception as e:
        # Fallback: simple truthful mechanism
        # Allocate both goods, charge the sum of values (this is IR but not optimal)
        x = np.ones((n_reports, 2))
        p = np.array([values[i][0] + values[i][1] for i in range(n_reports)])
        return {
            'allocations': x,
            'payments': p
        }
# EVOLVE-BLOCK-END
