import numpy as np
from scipy.optimize import linprog


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Uses a linear programming approach over a fixed sparse support grid.
    The objective is the exact probability P computed as a linear function of the weights.
    """
    # Define a sparse support grid with non-negative values
    # We want values that cover small (0, 1), medium, and large (M) ranges
    # The 'large' value M acts as a high threshold for X4
    # Small values 0 and 1 help minimize X1+X2+X3 when X4 is small
    
    # Support: 0, 1, and a large value M
    # M needs to be large enough that:
    # - When X4 = M, the condition X1+X2+X3 < 2*M is easily satisfied for small X1,X2,X3
    # - When X4 = 0 or 1, including M in {X1,X2,X3} makes the LHS >= M >= 2, so those cases fail
    # Let's use M = 10 (sufficiently large relative to 0 and 1)
    
    support = np.array([0.0, 1.0, 10.0], dtype=np.float64)
    n = len(support)
    
    # Precompute the coefficient matrix for the linear objective
    # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
    # This is actually a quartic form, not linear in weights!
    # Wait - the task says "computed exactly in float64" with the formula.
    # But the formula has w_k * w_l * (sum of w_i * w_j ...), which is degree 4 in weights.
    # So it's NOT linear. The directive says to use linprog, but the objective isn't linear.
    
    # Let me re-read: "Formulate the problem as a linear program where the objective function 
    # coefficients are the exact probabilities of the four independent draws summing to 
    # specific inequality outcomes"
    
    # Hmm, for fixed support points, P as a function of weights is a homogeneous polynomial 
    # of degree 4. It's not linear. So linprog won't directly work.
    
    # However, we can use scipy.optimize.minimize to maximize P over the simplex.
    # Or we can use a different approach.
    
    # Actually, let me just use numerical optimization over the simplex for the 3-point support.
    # This will find the optimal weights for support {0, 1, 10}.
    
    def compute_P(w):
        """Compute P[X1+X2+X3 < 2*X4] for i.i.d. draws from measure with weights w on support."""
        # w is a 1D array of length n
        # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
        # But wait, this formula doesn't look right dimensionally. Let me re-read.
        # "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
        # This has w_k*w_l*w_i*w_j which is degree 4. Yes.
        # But the indices: k and l are for X4 and... wait.
        # Let me re-read the task contract more carefully.
        # "P = Prob[X1 + X2 + X3 < 2*X4]"
        # "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
        # This seems to have only 4 indices but 4 variables X1,X2,X3,X4.
        # I think k corresponds to X4's value (s_k), and l corresponds to... hmm.
        # Actually I think there might be a typo or I'm misreading. Let me just compute P directly.
        
        # Direct computation: P = P(X1 + X2 + X3 < 2*X4)
        # = sum over all (i,j,m,k) of w_i * w_j * w_m * w_k * I(s_i + s_j + s_m < 2*s_k)
        
        P = 0.0
        for i in range(n):
            for j in range(n):
                for m in range(n):
                    for k in range(n):
                        if support[i] + support[j] + support[m] < 2.0 * support[k]:
                            P += w[i] * w[j] * w[m] * w[k]
        return P
    
    from scipy.optimize import minimize
    
    # Maximize P subject to sum(w) = 1, w >= 0
    # We'll minimize -P
    
    def neg_P(w):
        return -compute_P(w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    
    # Initial guess: equal weights
    w0 = np.ones(n) / n
    
    result = minimize(neg_P, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    w_opt = result.x
    # Clip to non-negative and renormalize
    w_opt = np.clip(w_opt, 0.0, None)
    total = np.sum(w_opt)
    if total < 1e-12:
        w_opt = np.ones(n) / n
    else:
        w_opt = w_opt / total
    
    return support, w_opt
