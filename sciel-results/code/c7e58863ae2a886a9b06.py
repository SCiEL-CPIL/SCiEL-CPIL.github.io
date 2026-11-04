import numpy as np
from scipy.optimize import minimize

N = 100
K = 5  # number of pairs of (1-segment, 0-segment)

def construct():
    """
    Construct the sequence of step heights using a differentiable optimization
    approach to minimize the maximum overlap (soft-max of cross-correlations).
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    # Parameters: lengths of 2K segments (alternating 1s and 0s)
    # x_i is constructed from these lengths using a smooth periodic square wave
    # Initial guess: equal lengths
    L0 = np.ones(2 * K) * (N / (2 * K))
    
    tau = 0.3  # softness of the step transition
    beta = 30.0  # sharpness of the soft-max
    
    def build_sequence(L):
        """
        Build the sequence x from segment lengths L.
        L has 2K entries: L[0], L[1], ..., L[2K-1]
        where even indices are 1-segments and odd indices are 0-segments.
        """
        # Cumulative boundaries
        B = np.cumsum(L)
        B = np.concatenate([[0.0], B])  # B[0]=0, B[1]=L[0], ..., B[2K]=sum(L)
        
        # Total length
        total = L.sum()
        
        # For each position i in [0, N), determine which segment it falls in
        # and whether it's a 1-segment or 0-segment
        i = np.arange(N, dtype=float)
        
        # Map i to the period [0, total)
        i_mod = i % total
        
        # For each i, find which segment it's in using searchsorted
        # B has 2K+1 entries, so segments are [B[j], B[j+1]) for j=0..2K-1
        # We need to handle the periodic wrap-around
        # Since i_mod is in [0, total), we can use searchsorted directly
        
        # Find the segment index for each i_mod
        # np.searchsorted(B, i_mod, side='right') - 1 gives the segment index
        # But we need to be careful with the boundaries
        seg_idx = np.searchsorted(B, i_mod, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, 2 * K - 1)
        
        # For each i, compute the smooth value
        # If seg_idx is even, it's a 1-segment; if odd, it's a 0-segment
        # We use a sigmoid transition at the boundaries
        
        x = np.zeros(N)
        for j in range(2 * K):
            seg_start = B[j]
            seg_end = B[j + 1]
            seg_len = L[j]
            
            # Positions in this segment
            mask = (seg_idx == j)
            if not np.any(mask):
                continue
            
            # Local coordinate within the segment
            # For positions that wrapped around, we need to handle carefully
            # Actually, since i_mod is in [0, total) and B is in [0, total],
            # the local coordinate is just i_mod - seg_start for those in this segment
            local = i_mod[mask] - seg_start
            
            # Smooth step: 
            # If j is even (1-segment): x = 1 in the middle, transitioning to 0 at boundaries
            # If j is odd (0-segment): x = 0 in the middle, transitioning to 1 at boundaries
            
            # Use a sigmoid centered at the segment boundaries
            # Left boundary: sigmoid((local - 0) / tau)
            # Right boundary: sigmoid((seg_len - local) / tau)
            
            left_sig = 1.0 / (1.0 + np.exp(-local / tau))
            right_sig = 1.0 / (1.0 + np.exp(-(seg_len - local) / tau))
            
            if j % 2 == 0:  # 1-segment
                # x should be 1 in the interior, 0 at the boundaries
                # x = left_sig * right_sig
                x[mask] = left_sig * right_sig
            else:  # 0-segment
                # x should be 0 in the interior, 1 at the boundaries
                # x = (1 - left_sig) * (1 - right_sig)
                x[mask] = (1.0 - left_sig) * (1.0 - right_sig)
        
        return x
    
    def loss_and_grad(L):
        """
        Compute the soft-max loss and its gradient with respect to L.
        """
        x = build_sequence(L)
        
        # Calculate C_k for all k
        C = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(x, -k)
            C[k] = np.sum(x * (1.0 - shifted))
        
        # Soft-max loss
        C_max = np.max(C[1:])
        exp_C = np.exp(beta * (C[1:] - C_max))
        sum_exp = np.sum(exp_C)
        loss = C_max + np.log(sum_exp) / beta
        
        # Compute gradient using finite differences (since analytical gradient is complex)
        grad = np.zeros(2 * K)
        eps = 1e-4
        for m in range(2 * K):
            L_plus = L.copy()
            L_plus[m] += eps
            x_plus = build_sequence(L_plus)
            
            C_plus = np.zeros(N)
            for k in range(1, N):
                shifted = np.roll(x_plus, -k)
                C_plus[k] = np.sum(x_plus * (1.0 - shifted))
            
            C_plus_max = np.max(C_plus[1:])
            exp_C_plus = np.exp(beta * (C_plus[1:] - C_plus_max))
            sum_exp_plus = np.sum(exp_C_plus)
            loss_plus = C_plus_max + np.log(sum_exp_plus) / beta
            
            grad[m] = (loss_plus - loss) / eps
        
        return loss, grad
    
    # Constraints
    def constraint_sum(L):
        return L.sum() - N
    
    def constraint_sum_even(L):
        return L[::2].sum() - N / 2.0
    
    constraints = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': lambda L: np.ones(2 * K)},
        {'type': 'eq', 'fun': constraint_sum_even, 'jac': lambda L: np.concatenate([np.ones(K), np.zeros(K)])}
    ]
    
    bounds = [(0.1, N - 0.1)] * (2 * K)
    
    res = minimize(
        loss_and_grad, 
        L0, 
        method='SLSQP', 
        jac=True, 
        constraints=constraints, 
        bounds=bounds, 
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    L_opt = res.x
    h = build_sequence(L_opt)
    
    # Final correction for exact sum
    h_sum = np.sum(h)
    if h_sum != N / 2.0:
        h += (N / 2.0 - h_sum) / N
    h = np.clip(h, 0.0, 1.0)
    
    # Ensure exact sum after clipping
    h_sum = np.sum(h)
    if h_sum != N / 2.0:
        # Distribute the difference evenly
        diff = N / 2.0 - h_sum
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    
    return h
