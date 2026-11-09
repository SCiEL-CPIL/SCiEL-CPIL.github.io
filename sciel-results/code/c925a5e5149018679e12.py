import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: the support of values (each good in {0,1,2,3})
        constraints: additional constraints (unused in this implementation)
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    # Parse the distribution to get probabilities for each of the 16 value vectors
    # distribution_spec should give us the probability of each (v1, v2) pair
    # Values are in {0,1,2,3} for each good
    
    # Extract probabilities from distribution_spec
    # Assuming distribution_spec is a dict or list mapping value vectors to probabilities
    probs = {}
    if isinstance(distribution_spec, dict):
        for key, prob in distribution_spec.items():
            if isinstance(key, (list, tuple)):
                v1, v2 = int(key[0]), int(key[1])
                probs[(v1, v2)] = float(prob)
            else:
                # Try to parse
                probs[key] = float(prob)
    elif isinstance(distribution_spec, (list, tuple)):
        # Assume it's a flat list of 16 probabilities in order (0,0),(0,1),...,(3,3)
        idx = 0
        for v1 in range(4):
            for v2 in range(4):
                if idx < len(distribution_spec):
                    probs[(v1, v2)] = float(distribution_spec[idx])
                    idx += 1
    else:
        # Default: uniform distribution
        for v1 in range(4):
            for v2 in range(4):
                probs[(v1, v2)] = 1.0 / 16.0
    
    # Ensure all 16 value vectors have probabilities
    for v1 in range(4):
        for v2 in range(4):
            if (v1, v2) not in probs:
                probs[(v1, v2)] = 0.0
    
    # Normalize probabilities to sum to 1
    total_prob = sum(probs.values())
    if total_prob > 0:
        for key in probs:
            probs[key] /= total_prob
    else:
        # Uniform default
        for v1 in range(4):
            for v2 in range(4):
                probs[(v1, v2)] = 1.0 / 16.0
    
    # Enumerate all monotone allocation rules
    # For each good, the allocation probability matrix must be non-decreasing in each dimension
    
    # We'll enumerate all possible monotone 4x4 matrices with values in [0,1]
    # But since the optimal solution will have allocations at extreme points,
    # we can restrict to {0,1} allocations first, then consider fractional if needed
    
    # Actually, for a single bidder, the optimal allocation is a step function
    # Let's enumerate all monotone binary allocation patterns
    
    best_revenue = -np.inf
    best_alloc1 = None
    best_alloc2 = None
    
    # For good 1: x1(v1,v2) must be non-decreasing in v1 for each v2
    # For good 2: x2(v1,v2) must be non-decreasing in v2 for each v1
    
    # Let's enumerate all possible monotone binary patterns for each good
    # A monotone binary 4x4 matrix is determined by a "threshold" structure
    
    # Generate all monotone binary 4x4 matrices
    def generate_monotone_binary_matrices():
        """Generate all 4x4 binary matrices that are non-decreasing in both rows and columns."""
        matrices = []
        # A matrix is monotone if M[i,j] <= M[i+1,j] and M[i,j] <= M[i,j+1]
        # We can enumerate all 2^16 = 65536 binary matrices and filter
        for bits in product([0, 1], repeat=16):
            M = np.array(bits).reshape(4, 4)
            # Check monotonicity in rows (non-decreasing in v1 for fixed v2)
            valid = True
            for j in range(4):
                for i in range(3):
                    if M[i, j] > M[i + 1, j]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            # Check monotonicity in columns (non-decreasing in v2 for fixed v1)
            for i in range(4):
                for j in range(3):
                    if M[i, j] > M[i, j + 1]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                matrices.append(M)
        return matrices
    
    # This might be too many, let's think differently
    # Actually, the number of monotone binary 4x4 matrices is the number of order ideals
    # in a 4x4 grid, which is C(8,4) = 70... no, it's more complex.
    # Let me just enumerate and filter - 65536 is manageable.
    
    mono_matrices = generate_monotone_binary_matrices()
    n_mono = len(mono_matrices)
    
    # For each pair of monotone allocation matrices (one for each good),
    # compute the expected revenue
    for idx1, alloc1 in enumerate(mono_matrices):
        for idx2, alloc2 in enumerate(mono_matrices):
            # Compute expected revenue for this allocation rule
            revenue = 0.0
            for v1 in range(4):
                for v2 in range(4):
                    p = probs[(v1, v2)]
                    if p == 0:
                        continue
                    x1 = alloc1[v1, v2]
                    x2 = alloc2[v1, v2]
                    # Payment: for a single bidder, the payment is determined by IC
                    # p(v1,v2) = v1*x1(v1,v2) + v2*x2(v1,v2) - U(v1,v2)
                    # where U is the utility, set to the minimum IC-feasible value
                    # The minimum utility for (v1,v2) is:
                    # U(v1,v2) = max over all (v1',v2') of [U(v1',v2') + (v1-v1')*x1(v1,v2) + (v2-v2')*x2(v1,v2)]
                    # But this is recursive. The standard approach:
                    # U(0,0) = 0 (IR)
                    # U(v1,v2) = U(v1-1,v2) + x1(v1,v2)*(1) + U(v1,v2-1) - U(v1-1,v2-1) + x2(v1,v2)*(1)
                    # Wait, let me use the proper formula.
                    # For a single bidder with additive values, the payment is:
                    # p(v1,v2) = v1*x1(v1,v2) + v2*x2(v1,v2) - U(v1,v2)
                    # where U(v1,v2) = max_{v1'<=v1, v2'<=v2} [sum of marginal utilities]
                    # Actually, the correct formula for the minimum utility satisfying IC:
                    # U(v1,v2) >= U(v1',v2) + (v1-v1')*x1(v1,v2) for all v1' <= v1
                    # U(v1,v2) >= U(v1,v2') + (v2-v2')*x2(v1,v2) for all v2' <= v2
                    # With U(0,0) = 0
                    # The tightest constraint gives:
                    # U(v1,v2) = max(
                    #   max_{v1' in 0..v1} [U(v1',v2) + (v1-v1')*x1(v1,v2)],
                    #   max_{v2' in 0..v2} [U(v1,v2') + (v2-v2')*x2(v1,v2)]
                    # )
                    # This is still recursive. Let me compute U for all value vectors first.
                    pass
            
            # Let me restructure: compute U for all value vectors, then revenue
            # This is getting complex in the inner loop. Let me precompute.
            pass
    
    # Let me restructure the computation
    best_revenue = -np.inf
    best_alloc1 = None
    best_alloc2 = None
    
    for alloc1 in mono_matrices:
        for alloc2 in mono_matrices:
            # Compute utility U(v1,v2) for all value vectors
            # U(0,0) = 0
            # For other vectors, U is the minimum value satisfying IC
            # We compute U in increasing order of v1 and v2
            
            U = np.zeros((4, 4))
            feasible = True
            
            for v1 in range(4):
                for v2 in range(4):
                    if v1 == 0 and v2 == 0:
                        U[v1, v2] = 0.0
                        continue
                    
                    # Compute the minimum U(v1,v2) satisfying IC
                    min_u = 0.0  # IR: U >= 0
                    
                    # IC from varying v1 (fix v2)
                    for v1_prime in range(v1):
                        # U(v1,v2) >= U(v1_prime, v2) + (v1 - v1_prime) * x1(v1,v2)
                        # But we need x1 at (v1,v2) for the constraint from v1_prime
                        # The IC constraint is: U(v1,v2) - v1*x1(v1,v2) - v2*x2(v1,v2) >= 
                        #   U(v1_prime,v2) - v1_prime*x1(v1_prime,v2) - v2*x2(v1_prime,v2) + 
                        #   (v1 - v1_prime)*x1(v1,v2) ... this is getting confusing.
                        # Let me use the standard formula.
                        pass
                    break
                break
            break
    
    # Let me use a cleaner formulation. For a single bidder, the payment is:
    # p(v) = v·x(v) - U(v)
    # where U(v) is the utility, and IC requires:
    # U(v) >= U(v') + (v - v')·x(v) for all v'
    # The minimum U(v) satisfying this with U(0,0)=0 is:
    # U(v) = max_{v' <= v} [sum of marginal terms]
    # For two goods, this becomes a 2D dynamic programming.
    
    # Actually, the standard result is that for a single bidder, the optimal mechanism
    # maximizes expected revenue, and the allocation is monotone. The payment is:
    # p(v1,v2) = v1*x1(v1,v2) + v2*x2(v1,v2) - U(v1,v2)
    # where U(v1,v2) = max(
    #   max_{v1' in [0,v1]} [U(v1',v2) + (v1-v1')*x1(v1,v2)],
    #   max_{v2' in [0,v2]} [U(v1,v2') + (v2-v2')*x2(v1,v2)]
    # )
    # with U(0,0) = 0.
    
    # Let me implement this properly.
    
    def compute_revenue(alloc1, alloc2, probs):
        """Compute expected revenue for given allocation matrices."""
        U = np.zeros((4, 4))
        
        for v1 in range(4):
            for v2 in range(4):
                if v1 == 0 and v2 == 0:
                    U[v1, v2] = 0.0
                    continue
                
                # Compute minimum U(v1,v2) satisfying IC
                min_u = 0.0  # IR constraint
                
                # IC constraints from v1 direction (varying v1, fixed v2)
                for v1_prime in range(v1):
                    # U(v1,v2) >= U(v1_prime, v2) + (v1 - v1_prime) * x1(v1,v2)
                    constraint = U[v1_prime, v2] + (v1 - v1_prime) * alloc1[v1, v2]
                    min_u = max(min_u, constraint)
                
                # IC constraints from v2 direction (fixed v1, varying v2)
                for v2_prime in range(v2):
                    # U(v1,v2) >= U(v1, v2_prime) + (v2 - v2_prime) * x2(v1,v2)
                    constraint = U[v1, v2_prime] + (v2 - v2_prime) * alloc2[v1, v2]
                    min_u = max(min_u, constraint)
                
                U[v1, v2] = min_u
        
        # Compute expected revenue
        revenue = 0.0
        for v1 in range(4):
            for v2 in range(4):
                p = probs[(v1, v2)]
                if p == 0:
                    continue
                x1 = alloc1[v1, v2]
                x2 = alloc2[v1, v2]
                payment = v1 * x1 + v2 * x2 - U[v1, v2]
                # Check nonnegative payment
                if payment < -1e-10:
                    return -np.inf  # Infeasible
                revenue += p * payment
        
        return revenue
    
    for alloc1 in mono_matrices:
        for alloc2 in mono_matrices:
            rev = compute_revenue(alloc1, alloc2, probs)
            if rev > best_revenue:
                best_revenue = rev
                best_alloc1 = alloc1
                best_alloc2 = alloc2
    
    # Now compute the final payments
    U = np.zeros((4, 4))
    for v1 in range(4):
        for v2 in range(4):
            if v1 == 0 and v2 == 0:
                U[v1, v2] = 0.0
                continue
            
            min_u = 0.0
            for v1_prime in range(v1):
                constraint = U[v1_prime, v2] + (v1 - v1_prime) * best_alloc1[v1, v2]
                min_u = max(min_u, constraint)
            
            for v2_prime in range(v2):
                constraint = U[v1, v2_prime] + (v2 - v2_prime) * best_alloc2[v1, v2]
                min_u = max(min_u, constraint)
            
            U[v1, v2] = min_u
    
    # Build the output arrays
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    idx = 0
    for v1 in range(4):
        for v2 in range(4):
            allocations[idx, 0] = best_alloc1[v1, v2]
            allocations[idx, 1] = best_alloc2[v1, v2]
            payments[idx] = v1 * best_alloc1[v1, v2] + v2 * best_alloc2[v1, v2] - U[v1, v2]
            # Ensure nonnegative payment
            if payments[idx] < 0:
                payments[idx] = 0.0
            idx += 1
    
    return {
        'allocations': allocations,
        'payments': payments
    }
