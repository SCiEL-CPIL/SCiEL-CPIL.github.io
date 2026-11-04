"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import numpy as np
from scipy.optimize import linprog

N = 100
STEP = 2.0 / N  # width of each step


def _compute_overlap_matrix():
    """
    Compute the overlap matrix M where M[i,j] is the overlap between
    step i of h and step j of (1-h), i.e., integral of h(x) * (1-h(x - t_j)) dx
    over the full domain [0, 2].
    
    Actually, let's think more carefully.
    
    h is a step function on [0, 2] with N steps of width STEP.
    h(x) = h_i for x in [i*STEP, (i+1)*STEP).
    
    1-h is also a step function: (1-h)(x) = 1 - h_i for x in [i*STEP, (i+1)*STEP).
    
    We shift 1-h by t and compute the overlap:
    overlap(t) = integral_0^2 h(x) * (1-h)(x - t) dx
    
    Wait, actually the problem says "overlap of h with shifted 1-h". Let me reconsider.
    
    The overlap at shift t is: integral h(x) * (1-h)(x + t) dx (or x - t, depending on convention).
    
    Let's use: overlap(t) = integral_0^2 h(x) * (1-h)(x - t) dx
    
    Since both h and 1-h are step functions with N steps, the overlap at any shift t
    is a piecewise constant function of t (actually piecewise linear in general, but
    since both are step functions, the product is step-like).
    
    Actually, for step functions, the overlap integral as a function of shift is a
    piecewise constant function (the product of two step functions, when integrated,
    gives a function that changes only when edges align).
    
    Let me think about this differently. The shift t can be any value in [0, 2] (or 
    perhaps we should consider all shifts modulo the domain). 
    
    For the LP approach, we need to discretize the shifts. Let's use M shift positions.
    
    For a given shift t, the overlap is:
    overlap(t) = sum_i h_i * (1-h_{j(i,t)}) * width_of_overlap(i, j(i,t))
    
    where j(i,t) is the step index of (1-h) that overlaps with step i at shift t,
    and width_of_overlap is the measure of the intersection.
    
    This is getting complex. Let me use a simpler discretization: evaluate the overlap
    at a fine grid of shifts, and for each shift, express the overlap as a linear function
    of the h_i variables.
    
    For shift t, divide [0,2] into small intervals where both h and shifted (1-h) are
    constant. The overlap integral is then a sum over these intervals of h_i * (1-h_k) * length.
    
    But h_i * (1-h_k) is NOT linear in h! It's bilinear.
    
    Hmm, so the overlap is NOT linear in h. This means LP won't directly work on h.
    
    Let me reconsider. The overlap is:
    integral h(x) * (1 - h(x-t)) dx = integral h(x) dx - integral h(x)h(x-t) dx = 1 - autocorrelation(t)
    
    where autocorrelation(t) = integral h(x)h(x-t) dx.
    
    So minimizing the maximum overlap is equivalent to maximizing the minimum autocorrelation... no wait.
    
    overlap(t) = 1 - autocorrelation(t)
    
    We want to minimize max_t overlap(t) = 1 - min_t autocorrelation(t).
    
    So we want to maximize min_t autocorrelation(t).
    
    But autocorrelation(t) = integral h(x)h(x-t) dx is still quadratic in h.
    
    OK so this is a non-convex problem in general. The LP approach on the CDF might work differently.
    
    Let me reconsider the CDF formulation. If F is the CDF of h (i.e., F(x) = integral_0^x h(u) du),
    then F is a convex, piecewise linear function with F(0)=0, F(2)=1, and 0 <= F'(x) <= 1.
    
    The overlap might be expressible in terms of F and its shifts in a linear way.
    
    Actually, let me just use a direct numerical approach: parameterize h as N variables in [0,1]
    with sum constraint, and use scipy.optimize.minimize with a smooth approximation of the
    max overlap, or use a sequential approach.
    
    Given the 150s budget, let me try:
    1. Start with h = 0.5 everywhere (the seed)
    2. Use projected gradient or a direct optimization to reduce the max overlap
    3. Use a smooth max (log-sum-exp) as the objective
    
    Let me implement this.
    """
    pass


def _overlap_at_shift(h, t, N, STEP):
    """Compute overlap integral of h with shifted (1-h) at shift t."""
    # h is defined on [0, 2] with N steps
    # (1-h) shifted by t: (1-h)(x - t) for x in [0, 2]
    # This means we need (1-h) evaluated at x - t, which for x in [0,2] means x-t in [-t, 2-t]
    # We need to handle the wrap-around or boundary carefully.
    
    # For simplicity, let's assume the domain wraps or we only consider shifts where
    # the overlap is well-defined. Actually, for the Erdos problem, we likely consider
    # shifts t in [0, 2] and the functions are defined on [0, 2].
    
    # Let's compute the overlap numerically by sampling.
    # But for optimization, we need a differentiable version.
    
    # Alternative: compute overlap analytically for step functions.
    # For shift t, divide [0, 2] into subintervals where both h(x) and h(x-t) are constant.
    
    # h(x) is constant on [i*STEP, (i+1)*STEP) for i=0,...,N-1
    # h(x-t) is constant on [i*STEP + t, (i+1)*STEP + t) for i=0,...,N-1, intersected with [0,2]
    
    # The breakpoints are at i*STEP for i=0,...,N and at i*STEP + t for i such that these fall in [0,2].
    
    t = t % 2.0  # wrap shift to [0, 2)
    
    # Collect all breakpoints
    breakpoints = set([0.0, 2.0])
    for i in range(N + 1):
        breakpoints.add(i * STEP)
    for i in range(N + 1):
        bp = i * STEP + t
        if 0.0 < bp < 2.0:
            breakpoints.add(bp)
        bp2 = i * STEP + t - 2.0
        if 0.0 < bp2 < 2.0:
            breakpoints.add(bp2)
    
    breakpoints = sorted(breakpoints)
    
    overlap = 0.0
    for k in range(len(breakpoints) - 1):
        a = breakpoints[k]
        b = breakpoints[k + 1]
        if b <= a + 1e-15:
            continue
        # Midpoint of this subinterval
        mid = (a + b) / 2.0
        # h(mid)
        i_h = int(mid / STEP)
        if i_h >= N:
            i_h = N - 1
        if i_h < 0:
            i_h = 0
        h_val = h[i_h]
        
        # h(mid - t) -> this is (1-h) shifted
        x_shifted = mid - t
        # Wrap x_shifted to [0, 2)
        x_shifted = x_shifted % 2.0
        i_shifted = int(x_shifted / STEP)
        if i_shifted >= N:
            i_shifted = N - 1
        if i_shifted < 0:
            i_shifted = 0
        one_minus_h_val = 1.0 - h[i_shifted]
        
        overlap += h_val * one_minus_h_val * (b - a)
    
    return overlap


def _max_overlap(h, N, STEP, n_shifts=200):
    """Compute the maximum overlap over a grid of shifts."""
    shifts = np.linspace(0, 2, n_shifts, endpoint=False)
    overlaps = np.array([_overlap_at_shift(h, t, N, STEP) for t in shifts])
    return np.max(overlaps), shifts[np.argmax(overlaps)]


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    h = np.full(N, 0.5)
    
    # Evaluate initial max overlap
    max_ov, best_shift = _max_overlap(h, N, STEP, n_shifts=500)
    best_h = h.copy()
    best_max_ov = max_ov
    
    # Use a smooth approximation: log-sum-exp of overlaps
    # and do gradient descent with projection
    
    n_shifts = 500
    shifts = np.linspace(0, 2, n_shifts, endpoint=False)
    
    def smooth_max_overlap(h_vec, beta=50.0):
        """Compute smooth approximation of max overlap using log-sum-exp."""
        overlaps = np.array([_overlap_at_shift(h_vec, t, N, STEP) for t in shifts])
        # Log-sum-exp for smooth max
        max_ov = np.max(overlaps)
        if max_ov > 1e-10:
            log_sum_exp = max_ov + np.log(np.sum(np.exp(overlaps - max_ov)))
        else:
            log_sum_exp = 0.0
        return log_sum_exp / beta  # scale down
    
    # Since _overlap_at_shift is piecewise linear in h (for fixed t), we can compute
    # the gradient analytically.
    
    def overlap_and_grad(h_vec, t):
        """Compute overlap at shift t and its gradient w.r.t. h."""
        t_wrapped = t % 2.0
        breakpoints = set([0.0, 2.0])
        for i in range(N + 1):
            breakpoints.add(i * STEP)
        for i in range(N + 1):
            bp = i * STEP + t_wrapped
            if 0.0 < bp < 2.0:
                breakpoints.add(bp)
            bp2 = i * STEP + t_wrapped - 2.0
            if 0.0 < bp2 < 2.0:
                breakpoints.add(bp2)
        
        breakpoints = sorted(breakpoints)
        
        overlap = 0.0
        grad = np.zeros(N)
        
        for k in range(len(breakpoints) - 1):
            a = breakpoints[k]
            b = breakpoints[k + 1]
            if b <= a + 1e-15:
                continue
            mid = (a + b) / 2.0
            i_h = min(int(mid / STEP), N - 1)
            i_h = max(i_h, 0)
            
            x_shifted = (mid - t_wrapped) % 2.0
            i_shifted = min(int(x_shifted / STEP), N - 1)
            i_shifted = max(i_shifted, 0)
            
            width = b - a
            h_val = h_vec[i_h]
            one_minus_h_shifted = 1.0 - h_vec[i_shifted]
            
            overlap += h_val * one_minus_h_shifted * width
            
            # d(overlap)/d(h[i_h]) = one_minus_h_shifted * width
            grad[i_h] += one_minus_h_shifted * width
            # d(overlap)/d(h[i_shifted]) = -h_val * width
            grad[i_shifted] += -h_val * width
        
        return overlap, grad
    
    # Gradient descent with projection onto {h in [0,1]^N, sum(h) = N/2}
    lr = 0.01
    beta = 100.0
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() > deadline - 20.0:
            break
        
        # Compute smooth max overlap gradient
        overlaps = np.zeros(n_shifts)
        grads = np.zeros(N)
        
        for idx, t in enumerate(shifts):
            ov, g = overlap_and_grad(h, t)
            overlaps[idx] = ov
            grads += g
        
        # Smooth max: sum exp(beta * overlaps) / beta
        max_ov = np.max(overlaps)
        exps = np.exp(beta * (overlaps - max_ov))
        weights = exps / np.sum(exps)
        
        smooth_grad = np.zeros(N)
        for idx in range(n_shifts):
            smooth_grad += weights[idx] * grads[idx]
        
        # Gradient descent step
        h_new = h - lr * smooth_grad
        
        # Project onto {h in [0,1]^N, sum(h) = N/2}
        # Use a simple projection: clip and adjust
        h_new = np.clip(h_new, 0.0, 1.0)
        # Adjust to satisfy sum constraint
        current_sum = np.sum(h_new)
        target_sum = N / 2.0
        if current_sum > 0 and current_sum < N:
            h_new += (target_sum - current_sum) / N
            h_new = np.clip(h_new, 0.0, 1.0)
            # If clipping changed the sum, do a few more iterations
            for _ in range(10):
                current_sum = np.sum(h_new)
                if abs(current_sum - target_sum) < 1e-12:
                    break
                h_new += (target_sum - current_sum) / N
                h_new = np.clip(h_new, 0.0, 1.0)
        
        h = h_new
        
        # Evaluate
        if it % 50 == 0 or it == n_iterations - 1:
            max_ov_curr, _ = _max_overlap(h, N, STEP, n_shifts=500)
            if max_ov_curr < best_max_ov:
                best_max_ov = max_ov_curr
                best_h = h.copy()
        
        # Adaptive learning rate
        if it % 100 == 99:
            lr *= 0.95
    
    # Final polish: use scipy.optimize.minimize with the smooth objective
    if time.time() < deadline - 10.0:
        try:
            def objective(h_flat):
                h_vec = np.clip(h_flat, 0.0, 1.0)
                max_ov, _ = _max_overlap(h_vec, N, STEP, n_shifts=200)
                return max_ov
            
            # Use Nelder-Mead or L-BFGS-B with the smooth version
            def smooth_objective(h_flat):
                h_vec = np.clip(h_flat, 0.0, 1.0)
                overlaps = np.array([_overlap_at_shift(h_vec, t, N, STEP) for t in np.linspace(0, 2, 100, endpoint=False)])
                max_ov = np.max(overlaps)
                if max_ov > 1e-10:
                    return max_ov + np.log(np.sum(np.exp(overlaps - max_ov))) / 20.0
                return 0.0
            
            # Projected gradient with tighter tolerance
            h_polish = best_h.copy()
            lr_polish = 0.001
            for it in range(2000):
                if time.time() > deadline - 5.0:
                    break
                
                overlaps = np.zeros(200)
                grads = np.zeros(N)
                shifts_polish = np.linspace(0, 2, 200, endpoint=False)
                
                for idx, t in enumerate(shifts_polish):
                    ov, g = overlap_and_grad(h_polish, t)
                    overlaps[idx] = ov
                    grads += g
                
                max_ov = np.max(overlaps)
                exps = np.exp(50.0 * (overlaps - max_ov))
                weights = exps / np.sum(exps)
                
                smooth_grad = np.zeros(N)
                for idx in range(200):
                    smooth_grad += weights[idx] * grads[idx]
                
                h_polish = h_polish - lr_polish * smooth_grad
                h_polish = np.clip(h_polish, 0.0, 1.0)
                
                current_sum = np.sum(h_polish)
                target_sum = N / 2.0
                h_polish += (target_sum - current_sum) / N
                h_polish = np.clip(h_polish, 0.0, 1.0)
                
                for _ in range(5):
                    current_sum = np.sum(h_polish)
                    if abs(current_sum - target_sum) < 1e-12:
                        break
                    h_polish += (target_sum - current_sum) / N
                    h_polish = np.clip(h_polish, 0.0, 1.0)
                
                if it % 50 == 0:
                    max_ov_curr, _ = _max_overlap(h_polish, N, STEP, n_shifts=500)
                    if max_ov_curr < best_max_ov:
                        best_max_ov = max_ov_curr
                        best_h = h_polish.copy()
                
                lr_polish *= 0.99
            
            # Ensure final sum constraint is met exactly
            current_sum = np.sum(best_h)
            best_h += (N / 2.0 - current_sum) / N
            best_h = np.clip(best_h, 0.0, 1.0)
        except Exception:
            pass
    
    # Final check and return
    current_sum = np.sum(best_h)
    if abs(current_sum - N / 2.0) > 1e-10:
        best_h += (N / 2.0 - current_sum) / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h


# Precompute nothing at module level to keep import fast
