"""Tensor decomposition for 2x3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def _verify(U, V, W):
        """Verify if U, V, W decompose T exactly."""
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(D, T, atol=1e-9)

    def _construct_block_decomp(rank_per_block):
        """
        Construct a decomposition by splitting the p dimension into blocks of size m (3).
        Here p=9, m=3, so we have 3 blocks.
        Each block corresponds to multiplying A (2x3) by a 3x3 block of B.
        We use schoolbook for each 2x3x3 block, which has rank 2*3*3 = 18? 
        Wait, schoolbook for 2x3x3 is 2*3*3 = 18? No.
        Schoolbook for C = AB where A is n x m and B is m x p is n*m*p.
        So for 2x3x3, rank is 2*3*3 = 18.
        Wait, earlier I thought 9. Let's re-evaluate.
        C_{ik} = sum_j A_{ij} B_{jk}.
        There are n*p outputs. Each requires m multiplications? No, m products summed.
        The bilinear rank is the number of scalar multiplications needed.
        Schoolbook uses n*m*p multiplications.
        For 2x3x3: 2*3*3 = 18 multiplications.
        So rank is 18.
        Then for 2x3x9 (3 blocks of 3), rank is 3 * 18 = 54.
        This is the same as the total schoolbook rank.
        
        So block decomposition with schoolbook blocks gives 54.
        Is there a better block decomposition?
        If we can find a rank 7 algorithm for 2x2x2 (Strassen), can we use it for 2x3x3?
        2x3x3 can be viewed as 2x2x3 plus 2x1x3?
        Or 2x3x2 plus 2x3x1?
        
        Actually, let's look at the "rank 6" hint again.
        If R(2,3,3) = 6, then R(2,3,9) = 18.
        Is R(2,3,3) = 6 possible?
        Lower bound for R(n,m,p) is max(nm, mp, np).
        For 2x3x3: max(6, 9, 6) = 9.
        So R(2,3,3) >= 9.
        The hint said "optimal rank-6". This contradicts the lower bound max(nm, mp, np) = 9.
        Unless the hint is referring to a different tensor or I am misinterpreting "rank".
        Tensor rank lower bound is indeed max(nm, mp, np).
        For 2x3x3, nm=6, mp=9, np=6. Max is 9.
        So R(2,3,3) >= 9.
        So the "rank 6" hint is likely incorrect or refers to a different problem context (e.g. complex numbers? No, usually same bound).
        Or maybe it's a 2x2x3? No.
        
        So R(2,3,3) is at least 9.
        Schoolbook is 18.
        Is there a rank 9 or 10 algorithm for 2x3x3?
        If we use Strassen (rank 7 for 2x2x2), can we extend?
        2x3x3 multiplication:
        C = A B.
        We can partition A into [A1 A2] where A1 is 2x2, A2 is 2x1.
        B into [B1; B2] where B1 is 2x3, B2 is 1x3.
        C = A1 B1 + A2 B2.
        A1 B1 is 2x2x3.
        A2 B2 is 2x1x3.
        
        Rank(2x2x3) <= Rank(2x2x2) + Rank(2x2x1)?
        Rank(2x2x2) = 7 (Strassen).
        Rank(2x2x1) = 4 (Schoolbook, no better known? Lower bound max(4,2,2)=4).
        So Rank(2x2x3) <= 7 + 4 = 11?
        Or maybe Rank(2x2x3) is 8?
        
        Rank(2x1x3) = 3 (Schoolbook, lower bound max(2,3,2)=3).
        
        So total rank <= 11 + 3 = 14?
        If we can achieve rank 14 for 2x3x3, then 2x3x9 would be 3 * 14 = 42.
        This is better than 54.
        
        Let's try to construct a rank 14 decomposition for 2x3x3.
        Or even better, just use the known Strassen decomposition for the 2x2 part and schoolbook for the rest.
        
        However, constructing this manually is complex and error-prone.
        The safest improvement is to ensure we return the schoolbook 54 correctly and quickly, and then *try* to find a lower rank.
        But the current program *fails* to return 54 quickly if it gets stuck in optimization?
        No, it has a fallback.
        But the optimization loop takes time.
        If we skip optimization and just return 54, we get score 1.0.
        If we can find a valid 42 or 27, we get > 1.0.
        
        Let's check if there is a simple rank 27 decomposition.
        If we treat 2x3x9 as three independent 2x3x3 multiplications, and if R(2x3x3)=9, then total rank 27.
        But R(2x3x3) >= 9. Is it 9?
        Schoolbook is 18.
        If R(2x3x3) is 9, that's a huge improvement (factor of 2).
        Is it possible?
        C_{ik} = sum_j A_{ij} B_{jk}.
        This is a bilinear map.
        For 2x2x2, rank 7 < 8.
        For 2x3x3, maybe rank 12?
        
        Actually, there is a known result: R(2,3,3) = 9 is NOT possible?
        Wait, lower bound is 9.
        If R(2,3,3) = 9, then it's "sub-cubic" in a sense?
        
        Let's assume we can't easily find a better than 54 analytically without extensive search.
        The best "safe" improvement is to make the fallback 54 instantaneous and then use remaining time to search for better ranks.
        The current program spends most time in optimization.
        If we reverse the logic:
        1. Return 54 (instant).
        2. If time remains, try to optimize for lower ranks.
        
        But the function `construct` must return the result. It can't "return 54 then keep searching".
        So we must decide: do we gamble on finding a better rank, or do we guarantee 54?
        Current program gambles. If it fails, it returns 54.
        But if it takes 110s to fail, it's fine.
        However, if it finds a *bad* solution (error > 1e-8) for rank 27, it returns that?
        "return { "U": U_rounded, ... }, err_rounded"
        If err_rounded is large, the evaluator might mark it invalid?
        "an invalid result ... scores 0".
        So if the optimization converges to a local minimum with error 1e-5, and we return it, we score 0.
        The current code returns `U_rounded` even if `err_rounded` is not small, as long as it's the "best candidate".
        This is dangerous.
        
        Fix: Only return a solution if the error is effectively zero (e.g., < 1e-8).
        If no valid solution is found, return the schoolbook 54.
        
        This ensures we never score 0 due to invalid results from failed optimizations.
        And we always score at least 1.0 (rank 54).
        
        Additionally, we can add a quick constructive check for a known lower rank if one exists.
        But given the uncertainty, the primary fix is to **prevent returning invalid solutions**.
        
        Also, we can speed up the schoolbook construction.
        
        Let's modify the code to:
        1. Always have the schoolbook 54 ready.
        2. Run optimization.
        3. If optimization yields a valid (low error) solution, return it.
        4. Otherwise, return schoolbook 54.
        
        The current code has a flaw: `if best_err < 1e-10: ... return ... else: return ...` inside `_try_rank`.
        If `_try_rank` returns a result with high error, `construct` accepts it.
        We need to check validity in `construct` before accepting.
        
        Also, the optimization might be slow. We can reduce `n_restarts` or `max_iter` if we prioritize safety.
        But let's keep the search and just fix the validity check.
        
        One more thing: The prompt says "integer/half-integer entries".
        The schoolbook has 0s and 1s. Valid.
        Strassen has 1s, -1s. Valid.
        If optimization finds floats like 0.500001, rounding to 0.5 is good.
        
        Let's implement the safety check.
"""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def _compute_error(params, rank):
        """Compute the error for a given parameter vector."""
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U = params[0:size_U].reshape(n * m, rank)
        V = params[size_U:size_U + size_V].reshape(m * p, rank)
        W = params[size_U + size_V:size_U + size_V + size_W].reshape(p * n, rank)
        
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = D - T
        return np.sum(err * err)

    def _get_matrices(params, rank):
        """Extract U, V, W from parameter vector."""
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U = params[0:size_U].reshape(n * m, rank)
        V = params[size_U:size_U + size_V].reshape(m * p, rank)
        W = params[size_U + size_V:size_U + size_V + size_W].reshape(p * n, rank)
        return U, V, W

    def _try_rank(rank, n_restarts=5, max_iter=2000):
        """Attempt to find a decomposition of the given rank."""
        best_err = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + restart * 100 + rank)
            
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    _compute_error, 
                    params_init, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
                )
                
                if res.success or res.fun < best_err:
                    if res.fun < best_err:
                        best_err = res.fun
                        best_params = res.x
                        
            except Exception as e:
                continue
                
        if best_params is None:
            return None, np.inf
            
        U, V, W = _get_matrices(best_params, rank)
        
        # Check if error is small enough
        if best_err < 1e-10:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            D_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
            err_rounded = np.max(np.abs(D_rounded - T))
            
            if err_rounded < 1e-8:
                return {
                    "U": U_rounded,
                    "V": V_rounded,
                    "W": W_rounded
                }, best_err
            else:
                # If rounding fails, do NOT return this. Return None to signal failure.
                return None, err_rounded

        return None, best_err

    best_result = None
    best_rank = np.inf
    
    # Try ranks starting from a reasonable lower bound.
    # Lower bound for rank is max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # So try 27, 28, ...
    # We limit the search to avoid timeout.
    for rank in range(27, 40):
        if time.time() > deadline - 10:
            break
            
        result, err = _try_rank(rank, n_restarts=2, max_iter=500)
        
        if result is not None:
            best_result = result
            best_rank = rank
            break

    if best_result is not None:
        return {
            "U": best_result["U"].tolist(),
            "V": best_result["V"].tolist(),
            "W": best_result["W"].tolist()
        }

    # Fallback: Schoolbook decomposition (Rank 54)
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
