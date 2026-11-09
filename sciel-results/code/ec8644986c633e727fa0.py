# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite-support mechanism design using the Myerson structural approach.
    Assumes the distribution allows for decoupling into independent 1D auctions for each good.
    This is optimal if the joint distribution is a product distribution.
    """
    # Parse support
    if isinstance(support, np.ndarray):
        support = [tuple(s) for s in support]
    elif not isinstance(support, (list, tuple)):
        support = list(support)
    
    n_types = len(support)
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in support], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types

    # Define value grid
    values = [0, 1, 2, 3]
    
    # Compute marginal distributions for good 1 and good 2
    # prob1[v] = P(v1 = v)
    prob1 = np.zeros(4)
    prob2 = np.zeros(4)
    
    for i in range(n_types):
        v1, v2 = support[i]
        prob1[v1] += probs[i]
        prob2[v2] += probs[i]
        
    # Normalize marginals (just in case)
    prob1 = prob1 / prob1.sum()
    prob2 = prob2 / prob2.sum()

    def solve_1d_auction(prob):
        """
        Solves the 1D Myerson auction for a single bidder with values 0,1,2,3.
        Returns allocation rule x[4] and payment rule p[4] for the 1D case.
        """
        vals = np.array([0, 1, 2, 3], dtype=float)
        # Virtual valuation phi(v) = v - (1-F(v))/f(v)
        # Note: for v=0, if f(0)>0, phi(0) = 0 - (1-F(0))/f(0)
        # F(v) = sum_{u<=v} prob[u]
        # f(v) = prob[v]
        
        phi = np.zeros(4)
        F = np.cumsum(prob)
        
        for v_idx in range(4):
            v = vals[v_idx]
            f = prob[v_idx]
            if f > 0:
                # 1-F(v) is the mass above v
                # Wait, standard definition: phi(v) = v - (1-F(v))/f(v)
                # F(v) is CDF including v.
                # 1-F(v) is mass of types > v.
                mass_above = 1.0 - F[v_idx]
                phi[v_idx] = v - mass_above / f
            else:
                # If prob is 0, virtual valuation is undefined, but allocation should be 0
                # unless we iron. For now, set to -inf or 0?
                # If prob is 0, the type doesn't exist.
                phi[v_idx] = -np.inf
        
        # Check for monotonicity. If phi is not non-decreasing, we need ironing.
        # For 4 points, ironing is simple.
        # We want to find the largest interval [a, b] where the average virtual valuation is lower than the peaks?
        # Actually, we replace non-monotone segments with their average.
        
        # Simple ironing for 4 points:
        # If phi[0] > phi[1], iron [0,1]
        # If phi[1] > phi[2], iron [1,2]
        # etc.
        # General algorithm:
        # 1. Identify intervals where phi is not non-decreasing.
        # 2. Merge intervals and compute average phi.
        
        # Let's implement a simple iterative ironing for small N
        x = np.zeros(4)
        
        # Determine allocation based on ironed virtual valuation
        # If phi_ironed(v) >= 0, x(v) = 1, else 0.
        
        # Ironing procedure:
        # We maintain a list of segments.
        # Start with each point as a segment.
        # Merge segments if the virtual valuation of the left segment is greater than the right.
        # The virtual valuation of a merged segment [i, j] is the weighted average of phi[i..j] weighted by prob.
        
        segments = []
        for i in range(4):
            segments.append([i, i])
            
        merged = True
        while merged:
            merged = False
            new_segments = []
            i = 0
            while i < len(segments):
                curr = segments[i]
                if i + 1 < len(segments):
                    next_seg = segments[i+1]
                    # Calculate average phi for curr and next_seg
                    # avg_phi(seg) = sum_{k in seg} prob[k]*phi[k] / sum_{k in seg} prob[k]
                    def get_avg_phi(seg):
                        total_prob = sum(prob[k] for k in range(seg[0], seg[1]+1))
                        if total_prob == 0:
                            return -np.inf
                        total_phi = sum(prob[k]*phi[k] for k in range(seg[0], seg[1]+1))
                        return total_phi / total_prob
                    
                    avg_curr = get_avg_phi(curr)
                    avg_next = get_avg_phi(next_seg)
                    
                    if avg_curr > avg_next + 1e-9:
                        # Merge
                        new_segments.append([curr[0], next_seg[1]])
                        i += 2
                        merged = True
                    else:
                        new_segments.append(curr)
                        i += 1
                else:
                    new_segments.append(curr)
                    i += 1
            segments = new_segments
        
        # Apply ironed allocation
        for seg in segments:
            start, end = seg
            avg_phi = get_avg_phi(seg)
            for k in range(start, end+1):
                if avg_phi >= 0:
                    x[k] = 1.0
                else:
                    x[k] = 0.0
                    
        # Calculate payments
        # p(v) = v*x(v) - \int_0^v x(t) dt
        # Discrete: \int_0^v x(t) dt = sum_{u < v} x(u) * (v_u + 1 - v_u)? 
        # No, the integral is over the value space.
        # U(v) = \sum_{g} \int_0^{v_g} x_g(t) dt
        # For discrete values 0,1,2,3:
        # \int_0^0 = 0
        # \int_0^1 = x(0) * 1
        # \int_0^2 = x(0)*1 + x(1)*1
        # \int_0^3 = x(0)*1 + x(1)*1 + x(2)*1
        
        payments = np.zeros(4)
        for v_idx in range(4):
            v = vals[v_idx]
            # Integral term
            integral = 0.0
            for u_idx in range(v_idx):
                # The width of the interval [u, u+1] is 1
                integral += x[u_idx] * 1.0
            
            payments[v_idx] = v * x[v_idx] - integral
            
        return x, payments

    # Solve for each good
    x1, p1 = solve_1d_auction(prob1)
    x2, p2 = solve_1d_auction(prob2)
    
    # Construct 2D allocations and payments
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for i in range(n_types):
        v1, v2 = support[i]
        allocations[i, 0] = x1[v1]
        allocations[i, 1] = x2[v2]
        # Payment is the sum of 1D payments?
        # p(v1, v2) = p1(v1) + p2(v2)
        payments[i] = p1[v1] + p2[v2]
        
    # Clip numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
