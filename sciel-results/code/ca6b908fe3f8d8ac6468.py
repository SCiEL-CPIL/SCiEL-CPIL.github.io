"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Default fallback: constant 0.5
    default_result = np.full(N, 0.5)
    
    # Formulate as LP: minimize t
    # Variables: h[0..N-1], t
    # Constraints:
    #   0 <= h[i] <= 1
    #   sum h[i] = N/2
    #   For each shift s (1..N-1): sum_i h[i]*(1-h[i+s mod N]) <= t
    #   i.e., N/2 - sum_i h[i]*h[i+s mod N] <= t
    #   i.e., sum_i h[i]*h[i+s mod N] >= N/2 - t
    #
    # But this is quadratic! h[i]*h[i+s] is bilinear.
    #
    # Alternative: For binary-like solutions, we can think differently.
    # Actually, let's use a different approach: the overlap h[i]*(1-h[i+s])
    # is linear in h if we fix one, but jointly it's bilinear.
    #
    # Let's try a different formulation. Since we want to minimize max overlap,
    # and the problem is symmetric, let's try a structured approach.
    #
    # Key insight: If h takes values in {0, 1} (or near it), the problem becomes
    # combinatorial. With sum = N/2, we have N/2 ones and N/2 zeros.
    # The overlap at shift s is the number of positions where h[i]=1 and h[i+s]=0.
    #
    # For a binary sequence with N/2 ones, the overlap at shift s is:
    # count of (1,0) pairs at distance s.
    #
    # We want to minimize the maximum over s of this count.
    #
    # This is related to difference sets / balanced sequences.
    # A good construction: use a pattern based on quadratic residues or similar.
    
    # Let's try: for even N, use a repeating pattern
    # Pattern for N=100: try to construct a binary sequence with minimal max overlap
    
    # Approach: Use scipy to optimize a relaxed version
    # Variables: h[0..N-1], t
    # Minimize t
    # Subject to:
    #   sum h = N/2
    #   0 <= h[i] <= 1
    #   For each s in 1..N-1: sum_i (h[i] - h[i]*h[i+s]) <= t
    #
    # The constraint is bilinear. Let's linearize by noting that for the purpose
    # of an upper bound, we can use the fact that h[i]*(1-h[j]) <= h[i] when h[j]>=0
    # But that's too loose.
    #
    # Better approach: Use the fact that we can enumerate shifts and for each shift,
    # the constraint sum_i h[i]*(1-h[i+s]) <= t can be written as:
    # sum_i h[i] - sum_i h[i]*h[i+s] <= t
    # N/2 - sum_i h[i]*h[i+s] <= t
    #
    # This is still bilinear. Let's try a heuristic approach instead.
    
    # Heuristic: Start with a good structured pattern and refine
    # Try: alternating blocks or a pseudo-random balanced sequence
    
    # Let's try a simple structured approach: 
    # For N=100, use a pattern with period p that balances overlaps
    
    # Actually, let's try the LP with linearization:
    # Introduce auxiliary variables for products, or use a different formulation.
    
    # Simpler: Use scipy minimize with a penalty approach
    from scipy.optimize import minimize
    
    def objective_and_grad(x):
        # x[0:N] = h, x[N] = t
        h = x[:N]
        t = x[N]
        # Compute max overlap
        max_overlap = 0
        for s in range(1, N):
            # overlap at shift s: sum_i h[i]*(1-h[(i+s)%N])
            # Only count valid pairs (all N pairs for circular)
            overlap = np.sum(h * (1 - np.roll(h, -s)))
            if overlap > max_overlap:
                max_overlap = overlap
        # Objective: t + penalty for constraint violation
        # We want max_overlap <= t, so penalty if max_overlap > t
        penalty = max(0, max_overlap - t)
        return t + penalty**2
    
    # This is slow. Let's try a direct construction instead.
    
    # Known good construction: For the Erdős problem, sequences with
    # near-constant autocorrelation are optimal. 
    # 
    # Let's try: h[i] = 0.5 + a * cos(2*pi*k*i/N) for some k
    # This has structured autocorrelation.
    
    # Actually, the simplest improvement: use a binary sequence with good properties.
    # For N=100, let's try a sequence based on a known good pattern.
    
    # Try: m-sequence like pattern or just optimize numerically
    
    # Fast approach: Use the fact that for the LP relaxation with bilinear terms,
    # we can use an iterative linearization (successive linear programming).
    
    # Initialize
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = np.inf
    
    # Compute initial score
    def compute_score(h):
        max_overlap = 0
        for s in range(1, N):
            overlap = np.sum(h * (1 - np.roll(h, -s)))
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap
    
    best_score = compute_score(h)
    
    # Successive LP linearization
    # At each step, linearize the bilinear term h[i]*h[i+s] around current h
    # h[i]*h[i+s] ≈ h[i]*h0[i+s] + h[i+s]*h0[i] - h0[i]*h0[i+s]
    # So sum h[i]*(1-h[i+s]) ≈ sum h[i] - sum(h[i]*h0[i+s] + h[i+s]*h0[i] - h0[i]*h0[i+s])
    # = N/2 - sum(h[i]*h0[i+s]) - sum(h[i+s]*h0[i]) + sum(h0[i]*h0[i+s])
    # = N/2 - 2*sum(h[i]*h0[i+s]) + sum(h0[i]*h0[i+s])  (by symmetry of the sum over i)
    # Wait, sum_i h[i]*h0[i+s] and sum_i h[i+s]*h0[i] are the same sum (just reindexed).
    # So: ≈ N/2 - 2*sum_i h[i]*h0[(i+s)%N] + sum_i h0[i]*h0[(i+s)%N]
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        h0 = h.copy()
        
        # Build LP: minimize t
        # Variables: h[0..N-1], t -> total N+1 variables
        # Minimize: t
        c = np.zeros(N + 1)
        c[N] = 1.0  # minimize t
        
        # Inequality constraints: for each s, sum_i h[i]*(1-h[i+s]) <= t
        # Linearized: N/2 - 2*sum_i h[i]*h0[(i+s)%N] + C_s <= t
        # where C_s = sum_i h0[i]*h0[(i+s)%N]
        # Rearranged: -2*sum_i h[i]*h0[(i+s)%N] - t <= -N/2 - C_s
        # Or: 2*sum_i h[i]*h0[(i+s)%N] + t >= N/2 + C_s
        # In linprog form (A_ub @ x <= b_ub):
        # -2*sum_i h[i]*h0[(i+s)%N] - t <= -N/2 - C_s
        
        A_ub = np.zeros((N - 1, N + 1))
        b_ub = np.zeros(N - 1)
        
        for idx, s in enumerate(range(1, N)):
            # Coefficients for h[i]: -2*h0[(i+s)%N]
            for i in range(N):
                A_ub[idx, i] = -2.0 * h0[(i + s) % N]
            A_ub[idx, N] = -1.0  # coefficient for t
            C_s = np.sum(h0 * np.roll(h0, -s))
            b_ub[idx] = -N / 2.0 - C_s
        
        # Equality constraint: sum h[i] = N/2
        A_eq = np.zeros((1, N + 1))
        A_eq[0, :N] = 1.0
        b_eq = np.array([N / 2.0])
        
        # Bounds: 0 <= h[i] <= 1, t >= 0
        bounds = [(0, 1)] * N + [(0, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
            if result.success:
                h_new = result.x[:N]
                h_new = np.clip(h_new, 0, 1)
                # Normalize to ensure sum = N/2
                h_new = h_new * (N / 2.0) / np.sum(h_new)
                h_new = np.clip(h_new, 0, 1)
                
                score_new = compute_score(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
                h = h_new
            else:
                break
        except Exception:
            break
    
    if time.time() > deadline:
        return best_h
    
    return best_h
# EVOLVE-BLOCK-END
