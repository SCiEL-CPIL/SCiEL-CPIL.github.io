import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def find_sidon_set(N, K, attempts=1000):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums:
                        conflict = True
                        break
                    if sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Initial Best: Uniform
    best_h = np.ones(N) / N
    best_ratio = np.sum(best_h**2) / (np.sum(best_h) * np.max(np.convolve(best_h, best_h)))

    # 2. Find best Sidon set (Sparse structure)
    best_sidon_h = None
    best_sidon_k = 0
    for K in range(12, 4, -1):
        if time.time() > deadline - 40.0:
            break
        attempts = 200 if K > 8 else 500
        S = find_sidon_set(N, K, attempts)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            best_sidon_h = h
            best_sidon_k = K
            break

    # 3. Define Optimization Functions
    # We want to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Subject to sum(h) = 1, h >= 0
    # Equivalent to minimizing M = ||h*h||_inf subject to ||h||_2^2 = 1 (and sum(h)=1 is redundant if we normalize, but let's keep it for stability)
    # Actually, let's fix ||h||_2^2 = 1 and minimize M.
    # Then R = 1 / (||h||_1 * M).
    # Since sum(h)=1 is a constraint, ||h||_1 = 1.
    # So we minimize M subject to sum(h)=1 and ||h||_2^2 = 1?
    # No, ||h||_2^2 is fixed to 1 in the objective ratio numerator.
    # Let's parameterize h such that ||h||_2 = 1.
    # Then we want to minimize M = max_k (h*h)_k.
    # And we must ensure sum(h) = 1?
    # If we fix ||h||_2=1, sum(h) is not fixed.
    # The ratio is R = 1 / (||h||_1 * M).
    # So we want to minimize ||h||_1 * M.
    # This is a non-smooth objective (product of norms).
    #
    # Alternative formulation:
    # Minimize M subject to:
    # 1. sum(h) = 1
    # 2. ||h||_2^2 >= 1/M  (This is tricky)
    #
    # Let's stick to the previous approach which worked:
    # Minimize M subject to:
    # 1. sum(h) = 1
    # 2. h_i >= 0
    # 3. M >= (h*h)_k for all k
    #
    # This finds the h that minimizes the peak of the autoconvolution for a fixed L1 norm.
    # The ratio is then R = ||h||_2^2 / M.
    # Note: If we minimize M, we might drive ||h||_2^2 to be small?
    # For fixed L1, ||h||_2^2 is minimized when h is uniform.
    # If h is uniform, M is also minimized.
    # The ratio R = ||h||_2^2 / M.
    # For uniform h: R = (1/N) / (1/N) = 1.
    # If we make h sparser, ||h||_2^2 increases. M also increases?
    # For Sidon set: ||h||_2^2 = 1/K. M = 2/K^2. R = (1/K) / (2/K^2) = K/2.
    # So we want to maximize R.
    # Minimizing M alone might lead to uniform (R=1).
    # We need to balance ||h||_2^2 and M.
    #
    # Let's maximize R directly using SLSQP?
    # Objective: -R(h)
    # Constraints: sum(h)=1, h>=0.
    # R(h) = sum(h^2) / max_k (h*h)_k.
    # This is non-smooth due to max.
    #
    # Let's use the auxiliary variable M again, but optimize for R.
    # Maximize sum(h^2) / M
    # Subject to:
    # M >= (h*h)_k
    # sum(h) = 1
    # h >= 0
    #
    # This is a non-linear fractional program.
    # Let y = sum(h^2). Maximize y/M.
    # Equivalent to maximizing y subject to M*y <= 1? No.
    #
    # Let's try a different variable change.
    # Let h = x / ||x||_2. Then ||h||_2 = 1.
    # R = 1 / (||h||_1 * M_h).
    # M_h = ||h*h||_inf = || (x/||x||_2) * (x/||x||_2) ||_inf = M_x / ||x||_2^2.
    # ||h||_1 = ||x||_1 / ||x||_2.
    # R = 1 / ( (||x||_1 / ||x||_2) * (M_x / ||x||_2^2) ) = ||x||_2^3 / (||x||_1 * M_x).
    # This looks complicated.
    #
    # Let's go back to minimizing M subject to sum(h)=1 and h>=0, but add a penalty or constraint to encourage sparsity?
    # Or simply run SLSQP on -R(h) with a smooth approximation of max?
    #
    # Let's try the "Minimize M" approach but start from a good initial guess (Sidon).
    # If we start from a Sidon set, SLSQP might refine the weights to improve R.
    #
    # Wait, if we minimize M, we move towards uniformity (lower M).
    # But we want higher R.
    # R = ||h||_2^2 / M.
    # If we minimize M, we decrease the denominator. But we might also decrease the numerator ||h||_2^2.
    # For uniform, R=1. For Sidon, R=K/2.
    # So minimizing M is the WRONG direction if we start from uniform.
    # We should MAXIMIZE ||h||_2^2 subject to M <= 1?
    # If we fix M=1, maximize ||h||_2^2.
    # Then R = ||h||_2^2.
    # Constraints:
    # 1. sum(h) = 1
    # 2. h >= 0
    # 3. (h*h)_k <= 1 for all k
    #
    # This is a convex optimization problem!
    # Maximize sum(h_i^2) is NOT convex. It's concave.
    # Maximizing a concave function is not convex optimization.
    # However, minimizing sum(h_i^2) is convex.
    #
    # Let's look at the dual?
    # Or just use SLSQP to maximize sum(h^2) subject to (h*h)_k <= 1.
    # SLSQP can handle non-convex objectives.
    #
    # Let's try this:
    # Maximize f(h) = sum(h^2)
    # Subject to:
    # sum(h) = 1
    # h >= 0
    # (h*h)_k <= 1 for all k
    #
    # Initial guess: Sidon set scaled so that max autoconv is 1.
    # For Sidon set of size K, max autoconv is 2/K^2 (if weights are 1/K).
    # If we scale h by alpha, max autoconv becomes alpha^2 * 2/K^2.
    # We want alpha^2 * 2/K^2 = 1 => alpha = K/sqrt(2).
    # Then sum(h) = K * (1/K) * alpha = alpha = K/sqrt(2).
    # We need sum(h)=1. So we scale by 1/alpha.
    # h_i = (1/K) * (1/alpha) = 1/(K * K/sqrt(2)) = sqrt(2)/K^2.
    # Let's check sum: K * sqrt(2)/K^2 = sqrt(2)/K. Not 1.
    #
    # Let's just use the ratio R directly in SLSQP with a smooth max approximation.
    # R(h) = sum(h^2) / max_k (h*h)_k.
    # Let M = max_k (h*h)_k.
    # Approximate M with soft-max: M_approx = log(sum(exp(c * (h*h)_k))) / c.
    # As c -> inf, M_approx -> M.
    #
    # Objective: -sum(h^2) / M_approx.
    #
    # Let's try this.

    def autoconv(h):
        return np.convolve(h, h)

    def calculate_ratio(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = autoconv(h)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / (l1 * peak)

    # Update best with Sidon result if found
    if best_sidon_h is not None:
        r_sidon = calculate_ratio(best_sidon_h)
        if r_sidon > best_ratio:
            best_ratio = r_sidon
            best_h = best_sidon_h.copy()

    # SLSQP Optimization
    # We will maximize R(h) directly.
    # To handle the max, we use a soft-max approximation in the objective?
    # No, SLSQP needs smooth constraints.
    # Let's use the auxiliary variable M again, but this time we want to maximize R = ||h||_2^2 / M.
    # This is equivalent to maximizing ||h||_2^2 * M^{-1}.
    # Let's define the objective as - (sum(h^2) / M).
    # And constraints:
    # M >= (h*h)_k for all k.
    # sum(h) = 1.
    # h >= 0.
    # M > 0.
    
    # This is a non-linear program.
    # Variables: h (N), M (1). Total N+1.
    
    def objective(x):
        h = x[:N]
        M = x[N]
        if M <= 0:
            return 1e10
        l2_sq = np.sum(h**2)
        return -l2_sq / M

    def objective_grad(x):
        h = x[:N]
        M = x[N]
        if M <= 0:
            return np.ones(N+1) * 1e10
        l2_sq = np.sum(h**2)
        grad = np.zeros(N+1)
        # d/dh_i (-sum(h^2)/M) = -2 h_i / M
        grad[:N] = -2.0 * h / M
        # d/dM (-sum(h^2)/M) = sum(h^2) / M^2
        grad[N] = l2_sq / (M * M)
        return grad

    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N+1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = autoconv(h)
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0
        return grad

    bounds = [(0.0, None)] * N + [(1e-6, None)]
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]

    # Initial guesses
    initial_guesses = []

    # 1. Current best (Uniform or Sidon)
    if best_h is not None:
        c_best = autoconv(best_h)
        M_best = np.max(c_best)
        x_best = np.concatenate([best_h, [M_best]])
        initial_guesses.append(x_best)
        
    # 2. Uniform
    h_uni = np.ones(N) / N
    c_uni = autoconv(h_uni)
    M_uni = np.max(c_uni)
    initial_guesses.append(np.concatenate([h_uni, [M_uni]]))

    # 3. Sidon sets (if found)
    if best_sidon_h is not None:
        c_sidon = autoconv(best_sidon_h)
        M_sidon = np.max(c_sidon)
        initial_guesses.append(np.concatenate([best_sidon_h, [M_sidon]]))

    # 4. Random sparse guesses
    for _ in range(30):
        if time.time() > deadline - 10.0:
            break
        K_rand = np.random.randint(5, 15)
        support = np.random.choice(N, K_rand, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K_rand
        c_rand = autoconv(h_rand)
        M_rand = np.max(c_rand)
        initial_guesses.append(np.concatenate([h_rand, [M_rand]]))

    # Run SLSQP on each guess
    for x0 in initial_guesses:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                jac=objective_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12, 'iprint': 0}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    continue
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = calculate_ratio(h_res)
                if ratio_res > best_ratio:
                    best_ratio = ratio_res
                    best_h = h_res.copy()
                    
        except Exception:
            continue

    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
