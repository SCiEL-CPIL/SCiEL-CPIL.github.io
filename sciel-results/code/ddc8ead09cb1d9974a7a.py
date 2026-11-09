# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill


def _compute_prob(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    # Build pairwise sums: for each pair (i,j), compute s_i + s_j
    # P = sum over k,l of w_k * w_l * (sum of w_i*w_j over (i,j) with s_i+s_j < 2*s_l - s_k)
    # Equivalent: P = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i + s_j < 2*s_l - s_k)
    # = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i + s_j + s_k < 2*s_l)
    
    # Vectorized: for each l, compute sum over i,j,k of w_i w_j w_k * I(s_i+s_j+s_k < 2*s_l)
    # Use outer products
    # s_i + s_j + s_k < 2*s_l
    # Let's compute the triple sum for each l
    
    P = 0.0
    for l in range(n):
        threshold = 2.0 * s[l]
        # For each (i,j,k), check s[i]+s[j]+s[k] < threshold
        # Use broadcasting
        s_i = s[:, None, None]
        s_j = s[None, :, None]
        s_k = s[None, None, :]
        triple_sums = s_i + s_j + s_k  # shape (n, n, n)
        mask = triple_sums < threshold
        w_i = w[:, None, None]
        w_j = w[None, :, None]
        w_k = w[None, None, :]
        prob_given_l = np.sum(mask * w_i * w_j * w_k)
        P += w[l] * prob_given_l
    return P


def _compute_prob_fast(support, weights):
    """Faster computation using convolution-like approach for small n."""
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / weights.sum()
    s = support
    
    # For 3-point support, we can enumerate all 27 combinations
    if n <= 4:
        P = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if s[i] + s[j] + s[k] < 2.0 * s[l]:
                            P += w[i] * w[j] * w[k] * w[l]
        return P
    
    # General case: use the formula
    P = 0.0
    for l in range(n):
        threshold = 2.0 * s[l]
        triple_sums = s[:, None, None] + s[None, :, None] + s[None, None, :]
        mask = triple_sums < threshold
        prob_given_l = np.sum(mask * w[:, None, None] * w[None, :, None] * w[None, None, :])
        P += w[l] * prob_given_l
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Search over 3-point supports {0, 1, M} with M >= 2, and also try
    2-point supports {0, 1} and other configurations.
    """
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    # First, try the simple 2-point {0, 1} with varying weights
    # P[X1+X2+X3 < 2*X4] for support {0,1}, weights (a, 1-a)
    # X4=0: need sum < 0, impossible. P=0
    # X4=1: need sum < 2, i.e., sum in {0, 1}
    #   sum=0: all 0s: a^3
    #   sum=1: one 1, two 0s: 3*a^2*(1-a)
    #   sum=2: two 1s, one 0: 3*a*(1-a)^2 -> NOT < 2
    #   sum=3: all 1s: (1-a)^3 -> NOT < 2
    # P = (1-a) * [a^3 + 3*a^2*(1-a)]
    # = (1-a) * a^2 * (a + 3 - 3a)
    # = (1-a) * a^2 * (3 - 2a)
    
    # Maximize f(a) = (1-a)*a^2*(3-2a) for a in [0,1]
    # f(a) = (1-a)(3a^2 - 2a^3) = 3a^2 - 2a^3 - 3a^3 + 2a^4 = 3a^2 - 5a^3 + 2a^4
    # f'(a) = 6a - 15a^2 + 8a^3
    # f'(a) = a(6 - 15a + 8a^2) = 0
    # 8a^2 - 15a + 6 = 0
    # a = (15 ± sqrt(225 - 192))/16 = (15 ± sqrt(33))/16
    # sqrt(33) ≈ 5.744
    # a1 = (15 - 5.744)/16 ≈ 0.5785
    # a2 = (15 + 5.744)/16 ≈ 1.30 (outside [0,1])
    
    a_opt = (15 - np.sqrt(33)) / 16
    prob_2pt = (1 - a_opt) * a_opt**2 * (3 - 2 * a_opt)
    
    if prob_2pt > best_prob:
        best_prob = prob_2pt
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([a_opt, 1 - a_opt])
    
    # Now try 3-point supports {0, 1, M} with M >= 2
    # Weights: (a, b, c) with a+b+c=1, a,b,c >= 0
    # P = P[X4=0]*P[sum<0|X4=0] + P[X4=1]*P[sum<2|X4=1] + P[X4=M]*P[sum<2M|X4=M]
    # = 0 + b * P[sum<2] + c * P[sum<2M]
    
    # For sum < 2 with values from {0,1,M}, M>=2:
    # Only sums using 0s and 1s with total < 2: (0,0,0) and permutations of (0,0,1)
    # P[sum<2] = a^3 + 3*a^2*b
    
    # For sum < 2M with M > 2:
    # Any combination without M: max sum = 3 < 2M (since M>2, 2M>4>3) ✓
    # One M: sum = M + x + y where x,y in {0,1}. Need M+x+y < 2M → x+y < M. Since M>2 and x+y<=2, always true.
    #   Prob: 3 * c * (a+b)^2 ... wait, need to be careful about which variable is M.
    #   Actually, sum = s_i + s_j + s_k. Exactly one of them is M: 3 choices for position, c for that position, (a+b) for each of the other two.
    #   But we also need to consider that the other two could be 0 or 1.
    #   P[exactly one M] = 3*c*(a+b)^2
    #   All of these have sum = M + 0,1,or 2 < 2M (since M>2)
    # Two M's: sum = 2M + x where x in {0,1}. Need 2M+x < 2M → x<0, impossible.
    # Three M's: 3M < 2M impossible.
    # P[sum<2M] = (a+b)^3 + 3*c*(a+b)^2
    #            = (a+b)^2 * [(a+b) + 3c]
    #            = (a+b)^2 * [a+b+3c]
    
    # So P = b*(a^3 + 3*a^2*b) + c*((a+b)^2*(a+b+3*c))
    # With a+b+c=1, c=1-a-b
    
    # Let's do a grid search over a and b (with c=1-a-b >= 0)
    # and also vary M (though for M>2, the formula doesn't depend on M!)
    
    # Wait, that's interesting. For M > 2, P doesn't depend on M. So any M > 2 works the same.
    # Let's verify: for M=3, P[sum<2M=6]. Max sum with one M is M+1+1=5<6 ✓. With two M's: 2M+0=6, not < 6. Correct.
    # For M=2.5, 2M=5. One M: 2.5+1+1=4.5<5 ✓. Two M's: 5+0=5, not <5. Correct.
    
    # So for M > 2, P = b*(a^3 + 3*a^2*b) + c*(a+b)^2*(a+b+3*c)
    # Let's maximize this over a,b >= 0, a+b <= 1.
    
    a_grid = np.linspace(0, 1, 200)
    b_grid = np.linspace(0, 1, 200)
    
    for a in a_grid:
        for b in b_grid:
            c = 1 - a - b
            if c < -1e-12:
                continue
            c = max(0, c)
            if a + b + c <= 0:
                continue
            
            # P = b*(a^3 + 3*a^2*b) + c*(a+b)^2*(a+b+3*c)
            term1 = b * (a**3 + 3 * a**2 * b)
            ab = a + b
            term2 = c * ab**2 * (ab + 3 * c)
            P = term1 + term2
            
            if P > best_prob:
                best_prob = P
                best_support = np.array([0.0, 1.0, 3.0])  # M=3 > 2
                best_weights = np.array([a, b, c])
    
    # Also try M=2 exactly (boundary case)
    # For M=2, 2M=4.
    # P[sum<2] same as before: a^3 + 3*a^2*b (only 0s and 1s with sum<2)
    # P[sum<4] with values from {0,1,2}:
    # Max sum = 6. Need sum < 4.
    # Enumerate: sum can be 0,1,2,3 (valid) or 4,5,6 (invalid)
    # This is more complex. Let's just compute it numerically for the best (a,b,c) found.
    
    # Verify the best solution
    if best_support is not None:
        verify_prob = _compute_prob_fast(best_support, best_weights)
        # The analytical formula should match for M>2
        # Let's also check if a different M helps (it shouldn't for M>2)
    
    # Also try some other 3-point supports: {0, M1, M2} or {0, 1, 2}
    # {0, 1, 2}: M=2 is boundary
    # Let's compute P for {0,1,2} with weights (a,b,c)
    # P[X4=0]: 0
    # P[X4=1]: P[sum<2] = a^3 + 3a^2b
    # P[X4=2]: P[sum<4]
    #   Sum of three from {0,1,2} < 4
    #   All combinations:
    #   (0,0,0):0 ✓, (0,0,1):1 ✓, (0,0,2):2 ✓, (0,1,1):2 ✓, (0,1,2):3 ✓, (0,2,2):4 ✗
    #   (1,1,1):3 ✓, (1,1,2):4 ✗, (1,2,2):5 ✗, (2,2,2):6 ✗
    #   Valid: (0,0,0),(0,0,1)x3,(0,0,2)x3,(0,1,1)x3,(0,1,2)x6,(1,1,1)
    #   P[sum<4] = a^3 + 3a^2b + 3a^2c + 3ab^2 + 6abc + b^3
    # P = b*(a^3+3a^2b) + c*(a^3+3a^2b+3a^2c+3ab^2+6abc+b^3)
    
    # Let's also grid search this
    for a in a_grid:
        for b in b_grid:
            c = 1 - a - b
            if c < -1e-12:
                continue
            c = max(0, c)
            
            term1 = b * (a**3 + 3 * a**2 * b)
            term2_inner = a**3 + 3*a**2*b + 3*a**2*c + 3*a*b**2 + 6*a*b*c + b**3
            term2 = c * term2_inner
            P = term1 + term2
            
            if P > best_prob:
                best_prob = P
                best_support = np.array([0.0, 1.0, 2.0])
                best_weights = np.array([a, b, c])
    
    # Verify final answer
    if best_support is not None:
        final_prob = _compute_prob_fast(best_support, best_weights)
        # Ensure validity
        if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
            raise ValueError("Invalid support")
        if np.any(best_support < -1e-9):
            raise ValueError("Negative support")
        if np.any(best_weights < -1e-9):
            raise ValueError("Negative weights")
        if np.sum(best_weights) <= 1e-9:
            raise ValueError("Sum of weights too small")
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
