"""Rewritten program: direct convex optimization over payment vector with IC-derived payments."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    # Parse the problem structure
    # One additive bidder, two goods, each value in {0, 1, 2, 3}
    # 16 possible value reports (4 x 4)
    
    values = [0, 1, 2, 3]
    # Each report is (v1, v2) where v1 is value for good 1, v2 for good 2
    # There are 16 possible types
    
    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            # Try to extract probabilities from the spec
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        # Default: uniform distribution over all 16 types
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Support: list of (v1, v2) pairs, or we can infer it
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                # support = (v1_support, v2_support)
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"
    
    # Map type index to (v1, v2)
    type_to_vals = {}
    for idx, (v1, v2) in enumerate(types):
        type_to_vals[idx] = (v1, v2)
    
    # For a single additive bidder with two goods:
    # Utility for type t=(v1,v2) when allocated (x1, x2) with payment p:
    # u(t) = v1*x1 + v2*x2 - p
    #
    # IC condition: For all types t and t':
    # u(t) >= u(t') when reporting t' instead of t
    # v1_t * x1(t') + v2_t * x2(t') - p(t') <= v1_t * x1(t) + v2_t * x2(t) - p(t)
    #
    # The unique IC payment function (up to a constant) is determined by the allocation rule.
    # For a single agent, the IC payment is:
    # p(t) = v1_t * x1(t) + v2_t * x2(t) - u(t)
    # where u(t) is the utility.
    #
    # Actually, for a single additive bidder, the mechanism is:
    # - Allocation rule x: T -> {0,1}^2 (or more generally [0,1]^2)
    # - Payment p(t) = v1(t)*x1(t) + v2(t)*x2(t) - u(t)
    #
    # IC requires that u(t) >= v1(t)*x1(t') + v2(t)*x2(t') - p(t') for all t, t'
    # Substituting p(t') = v1(t')*x1(t') + v2(t')*x2(t') - u(t'):
    # u(t) >= v1(t)*x1(t') + v2(t)*x2(t') - v1(t')*x1(t') - v2(t')*x2(t') + u(t')
    # u(t) - u(t') >= (v1(t) - v1(t'))*x1(t') + (v2(t) - v2(t'))*x2(t')
    #
    # This is a system of linear inequalities. The minimal payment (maximizing revenue)
    # corresponds to the maximal utility u(t) that satisfies IC and IR.
    #
    # IR: u(t) >= 0 for all t
    #
    # Revenue = sum_t p(t) * Pr(t) = sum_t (v1(t)*x1(t) + v2(t)*x2(t) - u(t)) * Pr(t)
    #
    # So maximizing revenue = maximizing sum_t (v1(t)*x1(t) + v2(t)*x2(t) - u(t)) * Pr(t)
    # which is equivalent to minimizing sum_t u(t) * Pr(t) subject to IC and IR constraints,
    # for a fixed allocation rule.
    #
    # But we can also optimize over the allocation rule. Let's set up the full LP.
    #
    # Variables: x1(t), x2(t) for t in T, and u(t) for t in T
    # But actually, we can eliminate u by expressing payments directly.
    #
    # Alternative approach: The payment p(t) is determined by the allocation rule and the
    # IC/IR conditions. For a single agent, the payment is:
    # p(t) = max_{t'} [v1(t)*x1(t') + v2(t)*x2(t') - v1(t')*x1(t') - v2(t')*x2(t') + p(t')]
    # This is recursive. 
    #
    # Better: Use the fact that for a single agent, the IC constraint is:
    # p(t) >= v1(t)*x1(t') + v2(t)*x2(t') - v1(t)*x1(t) - v2(t)*x2(t) + p(t) ... no
    #
    # Let me re-derive. Agent of type t reports t'. 
    # Utility when type t reports t': v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    # Utility when type t reports t: v1(t)*x1(t) + v2(t)*x2(t) - p(t)
    # IC: v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    # => p(t) <= v1(t)*x1(t) + v2(t)*x2(t) - v1(t)*x1(t') - v2(t)*x2(t') + p(t')
    #
    # This is a set of linear constraints on the payment vector p, given the allocation x.
    #
    # IR: v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= 0 => p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    #
    # Also p(t) >= 0 (nonnegative payments).
    #
    # Revenue = sum_t p(t) * Pr(t)
    #
    # So the problem is:
    # Maximize sum_t p(t) * Pr(t)
    # Subject to:
    # 1. p(t) - p(t') <= v1(t)*x1(t) + v2(t)*x2(t) - v1(t)*x1(t') - v2(t)*x2(t') for all t, t'
    # 2. p(t) >= 0 for all t
    # 3. x1(t), x2(t) in [0, 1] for all t (feasible allocations)
    # 4. IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t) for all t
    #
    # This is a linear program! Variables: x1(t), x2(t), p(t) for all t.
    #
    # Let's set this up.
    
    # Index mapping
    idx = {}
    for i, (v1, v2) in enumerate(types):
        idx[(v1, v2)] = i
    
    # Build the LP
    # Variables: x1_0, x1_1, ..., x1_15, x2_0, ..., x2_15, p_0, ..., p_15
    # Total: 16 + 16 + 16 = 48 variables
    
    n = 16
    # Variable indices:
    # x1: 0 to 15
    # x2: 16 to 31
    # p: 32 to 47
    
    # Objective: maximize sum p(t) * Pr(t)
    # linprog minimizes, so minimize -sum p(t) * Pr(t)
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # Constraint 1: IC constraints
    # p(t) - p(t') <= v1(t)*x1(t) + v2(t)*x2(t) - v1(t)*x1(t') - v2(t)*x2(t')
    # Rearranged: v1(t)*x1(t) + v2(t)*x2(t) - v1(t)*x1(t') - v2(t)*x2(t') - p(t) + p(t') >= 0
    # Or: -v1(t)*x1(t) - v2(t)*x2(t) + v1(t)*x1(t') + v2(t)*x2(t') + p(t) - p(t') <= 0
    
    A_rows = []
    b_rows = []
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            row = np.zeros(3 * n)
            # -v1(t)*x1(t) - v2(t)*x2(t) + v1(t)*x1(t') + v2(t)*x2(t') + p(t) - p(t')
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    # Constraint 2: IR
    # p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # => v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= 0
    # => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) <= 0
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_ub_row = 0
        b_rows.append(b_ub_row)
    
    # Constraint 3: Nonnegativity of payments is handled by bounds
    # Constraint 4: Allocations in [0, 1]
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, 3 * n))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Bounds:
    # x1(t), x2(t) in [0, 1]
    # p(t) >= 0
    bounds = []
    for t in range(n):
        bounds.append((0, 1))  # x1
    for t in range(n):
        bounds.append((0, 1))  # x2
    for t in range(n):
        bounds.append((0, None))  # p
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation, zero payments
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    
    # Extract allocations and payments
    x1 = x[0:16]
    x2 = x[16:32]
    p = x[32:48]
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    # Verify constraints and clip to ensure validity
    # Ensure allocations are in [0,1]
    allocations = np.clip(allocations, 0, 1)
    # Ensure payments are nonnegative
    payments = np.maximum(payments, 0)
    
    # Re-verify IC and IR after clipping
    # If violations exist, adjust
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        # IR check
        util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util < -1e-10:
            payments[t] = max(0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    # Final sanity check
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
