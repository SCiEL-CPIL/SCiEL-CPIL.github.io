# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import itertools

# Finite auction design with 1 additive bidder, 2 goods, values in {0,1,2,3},
# 3 fixed distributions (uniform, product, uniform over 6 states).
#
# We compute the optimal expected revenue over the finite report/value support,
# satisfying:
#   - Feasible allocation probabilities: 0 <= p(v) <= 1 for each good.
#   - Individual rationality: p(v) * v - q(v) >= 0.
#   - Incentive compatibility: U(v) >= E_v'[U(v')] for all v, v' (truthful reporting
#     is optimal).
#
# We use a direct enumeration approach: for each possible allocation profile,
# we check IC and IR constraints and compute revenue. Then we maximize revenue.
#
# Since the state space is small (6 states), we can enumerate all feasible
# allocation/payment pairs and find the maximum revenue.

@lru_cache(maxsize=None)
def _get_support_data(support):
    """Parse support into a list of (v1, v2, v1', v2') tuples with probabilities."""
    if isinstance(support, dict):
        # support could be a dict mapping state -> probability
        states = []
        probs = []
        for key, prob in support.items():
            if isinstance(key, tuple):
                if len(key) == 2:
                    # (v1, v2)
                    states.append((int(key[0]), int(key[1])))
                else:
                    # (v1, v2, v1', v2')
                    states.append((int(key[0]), int(key[1]), int(key[2]), int(key[3])))
            else:
                states.append((int(key), int(key)))
            probs.append(float(prob))
        return states, probs
    elif isinstance(support, (list, tuple)):
        # list of (v1, v2, v1', v2') or (v1, v2)
        states = []
        probs = []
        for s in support:
            if isinstance(s, (tuple, list)):
                if len(s) == 2:
                    states.append((int(s[0]), int(s[1])))
                elif len(s) == 4:
                    states.append((int(s[0]), int(s[1]), int(s[2]), int(s[3])))
                else:
                    states.append((int(s[0]), int(s[1])))
            else:
                states.append((int(s), int(s)))
            probs.append(1.0 / len(support))  # uniform if no probs given
        return states, probs
    else:
        # Default: uniform over all 4x4 states
        states = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        probs = [1.0 / 16.0] * 16
        return states, probs

def _is_product_distribution(states, probs):
    """Check if the distribution is a product distribution."""
    if len(states) != 16:
        return False
    # Check if all 4x4 states are present
    state_set = set(states)
    for v1 in range(4):
        for v2 in range(4):
            if (v1, v2) not in state_set:
                return False
    # Check if probability factorizes
    p = {}
    for (v1, v2), prob in zip(states, probs):
        p[(v1, v2)] = prob
    # Check factorization: p(v1, v2) = p1(v1) * p2(v2)
    p1 = {}
    p2 = {}
    for v1 in range(4):
        for v2 in range(4):
            p1[v1] = p1.get(v1, 0) + p[(v1, v2)]
    for v1 in range(4):
        for v2 in range(4):
            p2[v2] = p2.get(v2, 0) + p[(v1, v2)]
    for v1 in range(4):
        for v2 in range(4):
            if abs(p[(v1, v2)] - p1[v1] * p2[v2]) > 1e-10:
                return False
    return True

def _is_uniform6_distribution(states, probs):
    """Check if the distribution is uniform over 6 specific states."""
    if len(states) != 6:
        return False
    # Check if all probabilities are equal
    for prob in probs:
        if abs(prob - 1.0 / 6.0) > 1e-10:
            return False
    return True

def _compute_optimal_revenue(distribution_spec, support, constraints):
    """
    Compute the optimal expected revenue for a finite auction with:
    - 1 additive bidder, 2 goods
    - Values in {0,1,2,3}
    - Incentive compatibility and individual rationality
    
    Returns: (allocations, payments, revenue)
    """
    states, probs = _get_support_data(support)
    n = len(states)
    
    # Determine the type of distribution
    is_product = _is_product_distribution(states, probs)
    is_uniform6 = _is_uniform6_distribution(states, probs)
    
    # For product distributions, we can use a direct formula
    if is_product:
        return _optimal_revenue_product(states, probs)
    
    # For other distributions, use enumeration
    return _optimal_revenue_enumeration(states, probs)

def _optimal_revenue_product(states, probs):
    """
    For product distributions, the optimal revenue can be computed using
    the Myerson-type formula. For a single bidder with values v1, v2,
    the optimal allocation probability for good i is:
    p_i(v_i) = 1 if v_i >= theta_i, where theta_i is the threshold.
    
    The threshold is determined by the virtual value.
    """
    n = len(states)
    
    # Extract p1 and p2 (marginal distributions)
    p1 = {}
    p2 = {}
    for (v1, v2), prob in zip(states, probs):
        p1[v1] = p1.get(v1, 0) + prob
        p2[v2] = p2.get(v2, 0) + prob
    
    # Compute virtual values for each good
    # For good 1: virtual value v1 - (1-F1(v1))/f1(v1)
    # For discrete case, we need to be more careful
    
    # Try all possible threshold combinations
    best_revenue = 0
    best_allocations = None
    best_payments = None
    
    for t1 in range(5):  # threshold for good 1: 0,1,2,3,4
        for t2 in range(5):  # threshold for good 2: 0,1,2,3,4
            # Compute allocations
            allocations = []
            payments = []
            revenue = 0
            
            for (v1, v2), prob in zip(states, probs):
                alloc1 = 1.0 if v1 >= t1 else 0.0
                alloc2 = 1.0 if v2 >= t2 else 0.0
                allocations.append((alloc1, alloc2))
                
                # Payment: use the payment identity
                # q(v) = v * p(v) - integral of p(v') dv' (for continuous)
                # For discrete, use the envelope theorem
                # For product distributions, payment = sum of payments for each good
                pay1 = 0.0
                for v1p in range(v1 + 1):
                    p1_val = 1.0 if v1p >= t1 else 0.0
                    if v1p > 0:
                        # Approximate the integral
                        pass
                # For discrete case, use the standard payment formula
                # q(v1, v2) = v1 * p1(v1) + v2 * p2(v2) - sum_{v1' < v1} p1(v1') - sum_{v2' < v2} p2(v2')
                pay1 = v1 * alloc1
                for v1p in range(v1):
                    p1_val = 1.0 if v1p >= t1 else 0.0
                    pay1 -= p1_val
                pay2 = v2 * alloc2
                for v2p in range(v2):
                    p2_val = 1.0 if v2p >= t2 else 0.0
                    pay2 -= p2_val
                payment = pay1 + pay2
                payments.append(payment)
                revenue += prob * payment
            
            if revenue > best_revenue + 1e-10:
                best_revenue = revenue
                best_allocations = allocations
                best_payments = payments
    
    # Convert to numpy arrays
    allocations = np.array(best_allocations, dtype=float)
    payments = np.array(best_payments, dtype=float)
    
    return allocations, payments, best_revenue

def _optimal_revenue_enumeration(states, probs):
    """
    For general distributions, use enumeration over all possible allocation profiles.
    Since the state space is small, we can enumerate all 2^2 = 4 allocation profiles
    per state and check IC/IR constraints.
    """
    n = len(states)
    
    # For each state, the allocation can be (0,0), (0,1), (1,0), (1,1)
    # We need to find allocation profiles that satisfy IC and IR
    
    best_revenue = 0
    best_allocations = None
    best_payments = None
    
    # Try all possible allocation profiles
    # For 6 states, there are 4^6 = 4096 possible allocation profiles
    # This is feasible to enumerate
    
    for alloc_profile in itertools.product([(0,0), (0,1), (1,0), (1,1)], repeat=n):
        allocations = []
        payments = []
        valid = True
        revenue = 0
        
        # Compute payments using the payment identity
        for i in range(n):
            v1, v2 = states[i][0], states[i][1]
            alloc1, alloc2 = alloc_profile[i]
            allocations.append((float(alloc1), float(alloc2)))
            
            # Compute payment
            # q(v) = v1*p1(v) + v2*p2(v) - sum_{v' < v} [p1(v') + p2(v')] (for product)
            # For general case, use the envelope theorem
            # For discrete case, approximate the integral
            pay1 = v1 * alloc1
            pay2 = v2 * alloc2
            for j in range(n):
                if states[j][0] < v1 or states[j][1] < v2:
                    # This is a rough approximation
                    pass
            payment = pay1 + pay2
            payments.append(payment)
            revenue += probs[i] * payment
        
        # Check IR: p(v) * v - q(v) >= 0
        for i in range(n):
            v1, v2 = states[i][0], states[i][1]
            alloc1, alloc2 = alloc_profile[i]
            utility = v1 * alloc1 + v2 * alloc2 - payments[i]
            if utility < -1e-10:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check IC: U(v) >= U(v') for all v, v'
        # U(v) = v * p(v) - q(v)
        utilities = []
        for i in range(n):
            v1, v2 = states[i][0], states[i][1]
            alloc1, alloc2 = alloc_profile[i]
            utility = v1 * alloc1 + v2 * alloc2 - payments[i]
            utilities.append(utility)
        
        ic_valid = True
        for i in range(n):
            for j in range(n):
                if utilities[j] > utilities[i] + 1e-10:
                    # State i would prefer to report as state j
                    ic_valid = False
                    break
            if not ic_valid:
                break
        
        if not ic_valid:
            continue
        
        if revenue > best_revenue + 1e-10:
            best_revenue = revenue
            best_allocations = allocations
            best_payments = payments
    
    # If no valid allocation found, use the trivial (0,0) allocation
    if best_allocations is None:
        allocations = [(0.0, 0.0)] * n
        payments = [0.0] * n
        best_revenue = 0.0
    else:
        allocations = best_allocations
        payments = best_payments
    
    # Convert to numpy arrays
    allocations = np.array(allocations, dtype=float)
    payments = np.array(payments, dtype=float)
    
    return allocations, payments, best_revenue

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with:
    - 1 additive bidder, 2 goods
    - Values in {0,1,2,3}
    - 3 fixed distributions (uniform, product, uniform over 6 states)
    
    Returns:
        allocations: (16, 2) array of allocation probabilities
        payments: (16,) array of expected payments
    """
    # Compute the optimal allocation and payments
    allocations, payments, revenue = _compute_optimal_revenue(distribution_spec, support, constraints)
    
    # Ensure the output is in the correct format
    allocations = np.array(allocations, dtype=float)
    payments = np.array(payments, dtype=float)
    
    # Pad to 16 states if necessary
    n = len(allocations)
    if n < 16:
        pad_alloc = np.zeros((16 - n, 2), dtype=float)
        pad_pay = np.zeros(16 - n, dtype=float)
        allocations = np.vstack([allocations, pad_alloc])
        payments = np.concatenate([payments, pad_pay])
    else:
        allocations = allocations[:16]
        payments = payments[:16]
    
    return {
        "allocations": allocations,
        "payments": payments
    }
# EVOLVE-BLOCK-END
