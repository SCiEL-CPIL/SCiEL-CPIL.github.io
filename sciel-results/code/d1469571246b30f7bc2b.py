import numpy as np
from scipy.optimize import linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # We want to maximize L2^2 / M.
    # Fix L1 = 1. Maximize L2^2 / M.
    # This is non-convex.
    # However, the directive suggests minimizing M (peak autoconv) via LP.
    # Minimizing M subject to L1=1, h>=0 yields Uniform h.
    # Uniform h has M_min = 1/N (discrete).
    # L2^2 = 1/N.
    # Ratio = (1/N) / (1/N) = 1.
    # Raw Score = 1 / DELTA = 100.
    
    # Let's try to solve the LP:
    # Minimize P
    # Subject to:
    # 1. sum(h) = 1
    # 2. h >= 0
    # 3. P >= (h*h)_j for all j (linear constraints on h and P)
    
    # Variables: h_0 ... h_{N-1}, P
    # Total variables: N + 1
    
    # Objective: Minimize P (index N)
    c = np.zeros(N + 1)
    c[N] = 1.0
    
    # Constraints:
    # Equality: sum(h) = 1
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])
    
    # Inequality: P - (h*h)_j >= 0 for all j
    # (h*h)_j = sum_i h_i h_{j-i}
    # This is NOT linear in h!
    # The directive says "minimizing the peak autoconvolution, which is a linear function of the height vector".
    # This is FALSE. The autoconvolution is quadratic.
    # UNLESS the "height vector" refers to something else?
    # Or maybe the directive implies a relaxation?
    # "thereby transforming the ratio maximization into a standard linear programming problem"
    
    # If the directive is wrong about linearity, I should still try to follow the intent:
    # Find a distribution that minimizes the peak.
    # The minimum peak for L1=1 is achieved by Uniform.
    
    # So I will just return Uniform.
    # But wait, the current program (Sidon) also gives 100.
    # If I return Uniform, I get 100.
    # If I return Sidon, I get 100.
    
    # Is there any other interpretation?
    # Maybe "autoconvolution" in the LP context means something linear?
    # No.
    
    # I will return Uniform as it is the solution to "Minimize Peak Autoconvolution".
    h = np.ones(N) / N
    return list(h)
