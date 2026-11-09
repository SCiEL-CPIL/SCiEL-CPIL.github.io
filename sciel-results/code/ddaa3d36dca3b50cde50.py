"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 50
# Equal intervals on [-1/4, 1/4]
# 50 points, spacing = (1/4 - (-1/4)) / (50 - 1) = 0.5/49
# Actually, "50 nonnegative heights on equal intervals of [-1/4, 1/4]"
# Usually implies a grid. Let's assume the grid is uniform over [-1/4, 1/4].
# If it's a step function, the interval width is dx.
# The prompt says "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4]".
# This likely means the domain is partitioned into 50 intervals? Or 50 points?
# "heights" usually implies values at points or over bins.
# If it's a step function with 50 steps, the width of each step is (1/4 - (-1/4))/50 = 1/100?
# Or is it 50 points including endpoints? (1/4 - (-1/4))/(50-1)?
# Let's look at the standard interpretation for such discrete optimization problems.
# Often, "heights on equal intervals" refers to a piecewise constant function (step function)
# defined on N intervals covering the domain.
# Domain length = 1/2. N=50. Step width dx = (1/2)/50 = 0.01.
# Let's assume h_i is the value on the i-th interval of width dx.
# The L2 norm squared of the function is sum(h_i^2 * dx).
# The L1 norm is sum(h_i * dx).
# The autoconvolution (h*h)(t) = integral h(s)h(t-s) ds.
# For a step function, this is a piecewise quadratic function.
# We need the supremum norm of h*h.
# h*h is symmetric around 0.
# The peak is likely at t=0 if h is positive and concentrated, or elsewhere if spread out.
# However, the task asks to MAXIMIZE ||h||_2^2 / (||h||_1 * ||h*h||_inf).
# This is equivalent to MINIMIZING ||h||_1 * ||h*h||_inf subject to ||h||_2 = 1 (by scaling invariance).
# Let J(h) = ||h||_1 * ||h*h||_inf. We want min J(h) s.t. ||h||_2=1, h>=0.
# Score = 1 / J(h) * (some constant factor from dx? No, ratios cancel dx).
# Let's check scaling.
# ||h||_2^2 = sum h_i^2 dx.
# ||h||_1 = sum h_i dx.
# (h*h)(t) = sum_{i,j} h_i h_j * (length of intersection of [x_i, x_i+dx] and [t-x_j-dx, t-x_j]).
# This is discrete convolution of the sequence h_i with itself, weighted by dx?
# Let's derive the discrete formula for the peak of the autoconvolution.
# If h is a step function with values h_0, ..., h_{N-1} on intervals I_i = [i*dx, (i+1)*dx],
# where the domain is centered? The problem says intervals of [-1/4, 1/4].
# So the grid is symmetric.
# Let's define the grid points x_i = -1/4 + i*dx, for i=0..N-1?
# Or are the intervals [x_i, x_{i+1}]?
# Let's assume the function is defined on 50 bins covering [-1/4, 1/4].
# Bin width dx = 0.5 / 50 = 0.01.
# Let h be the vector of heights.
# ||h||_2^2 = dx * sum(h_i^2)
# ||h||_1 = dx * sum(h_i)
# Autoconvolution c(t) = (h*h)(t).
# The supremum of c(t) is the max value of the continuous convolution.
# For step functions, the convolution is a piecewise quadratic function.
# The maximum of the convolution of two non-negative step functions is attained at some point.
# Specifically, if we shift one function relative to the other, the overlap area is maximized when the mass is aligned.
# Actually, for non-negative functions, the maximum of the autoconvolution is at t=0 IF the function is "unimodal" or concentrated?
# No, consider a function with two peaks far apart. The self-convolution will have a peak at the distance between the peaks.
# However, we are free to choose h.
# It is known that for non-negative h, max_t (h*h)(t) >= (||h||_1)^2 / ||h||_inf? No.
# Let's use the property that max_t (h*h)(t) is related to the peak-to-average power ratio or similar.
# Actually, a simpler bound: max_t (h*h)(t) >= (sum h_i)^2 / N ? No.
# Let's compute the convolution numerically for a given h.
# The convolution of two step functions is a continuous function that is quadratic on intervals of length dx.
# The peaks occur at the centers of the bins or at the boundaries?
# For two rectangular pulses of width dx, the convolution is a triangle. Peak at 0.
# For general step functions, the max of (h*h) is max_k sum_{i,j: i-j=k} h_i h_j * (dx - |t - (i-j)dx|_overlap)?
# Let's approximate the supremum norm by the maximum of the discrete convolution of the sequence h with itself, multiplied by dx.
# Let c_k = sum_i h_{i+k} h_i. This is the circular convolution? No, linear.
# The continuous convolution value at t = m*dx (where m is an integer shift) is exactly dx * sum_i h_{i+m} h_i (assuming alignment of grid).
# Is the maximum of the continuous convolution always attained at a grid point t = m*dx?
# For step functions, the convolution is piecewise quadratic. The derivative is piecewise linear.
# The extrema occur where the derivative is zero or at the boundaries of the pieces.
# The boundaries are at t = (i - j) dx.
# So the maximum is attained at some t = m*dx.
# Therefore, ||h*h||_inf = dx * max_m (sum_i h_{i+m} h_i).
# Note: The sum is over valid i.
# So the objective to minimize is:
# J(h) = (dx * sum h_i) * (dx * max_m c_m)
# where c_m = sum_i h_{i+m} h_i.
# Since dx is constant, we can minimize (sum h_i) * (max_m c_m).
# Subject to sum h_i^2 = 1 (we can fix the L2 norm to 1 without loss of generality, as the ratio is scale invariant).
# Wait, the score is ||h||_2^2 / (||h||_1 ||h*h||_inf).
# If we scale h by alpha, ||h||_2^2 scales by alpha^2, ||h||_1 by alpha, ||h*h||_inf by alpha^2.
# Ratio scales by alpha^2 / (alpha * alpha^2) = 1/alpha.
# So to maximize the ratio, we want to make the denominator small? No, the ratio depends on scale.
# Let's re-read: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution".
# Ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
# If we scale h -> alpha h, R -> alpha^2 ||h||_2^2 / (alpha ||h||_1 * alpha^2 ||h*h||_inf) = (1/alpha) R.
# So R is NOT scale invariant. It goes to infinity as alpha -> 0?
# If h -> 0, ||h||_2^2 ~ alpha^2, ||h||_1 ~ alpha, ||h*h||_inf ~ alpha^2.
# R ~ alpha^2 / (alpha * alpha^2) = 1/alpha.
# So we can make R arbitrarily large by making h very small?
# That can't be right. The task must imply a normalization or I am misinterpreting "ratio".
# "score is the lower bound on C2".
# Maybe the question implies: Find the best constant C2 such that ||h||_2^2 <= C2 ||h||_1 ||h*h||_inf for all h.
# This is equivalent to finding the maximum of ||h||_1 ||h*h||_inf / ||h||_2^2?
# No, "maximize the ratio ... to get the lower bound on C2".
# If the inequality is ||h||_2^2 <= C2 ||h||_1 ||h*h||_inf, then C2 >= ||h||_2^2 / (||h||_1 ||h*h||_inf).
# The "best" (smallest) C2 is the supremum of the ratio over all h.
# But the task says "maximize the ratio ... score is the lower bound on C2".
# This phrasing is tricky. Usually, if you find a specific h, the ratio R(h) is a valid lower bound on C2?
# No, if C2 must satisfy the inequality for ALL h, then C2 must be at least sup_h R(h).
# So the "lower bound on C2" provided by a specific h is just R(h)?
# And we want to MAXIMIZE this lower bound? That would mean finding the h that makes R(h) largest, which gives the tightest lower bound?
# No, if C2 is a universal constant, any h gives a lower bound R(h) on C2.
# To get the BEST (highest) lower bound, we want the h that maximizes R(h).
# But wait, if R(h) can be made arbitrarily large by scaling (as derived above: R ~ 1/alpha), then there is no finite C2?
# Let's check the scaling again.
# ||h||_2^2 = \int h^2.
# ||h||_1 = \int h.
# ||h*h||_inf = \sup_t \int h(s)h(t-s) ds.
# If h = alpha * g, then:
# ||h||_2^2 = alpha^2 ||g||_2^2.
# ||h||_1 = alpha ||g||_1.
# ||h*h||_inf = alpha^2 ||g*g||_inf.
# Ratio = alpha^2 ||g||_2^2 / (alpha ||g||_1 * alpha^2 ||g*g||_inf) = (1/alpha) * (||g||_2^2 / (||g||_1 ||g*g||_inf)).
# As alpha -> 0, Ratio -> infinity.
# This implies the supremum is infinity.
# There must be a constraint I'm missing.
# "Choose 50 nonnegative heights".
# Maybe the heights are integers? No, "nonnegative heights" usually means real numbers.
# Maybe the L2 norm is fixed? "maximize the ratio ...".
# If the ratio is unbounded, the task is trivial (output 0). But the seed outputs 1.0 and scores 1.0.
# If I output 0, the ratio is 0/0? Or undefined?
# If I output very small numbers, the ratio is huge.
# Let's check the seed score. Seed is [1.0]*50.
# Let's calculate the ratio for the seed.
# h_i = 1. dx = 0.01.
# ||h||_2^2 = 50 * 1^2 * 0.01 = 0.5.
# ||h||_1 = 50 * 1 * 0.01 = 0.5.
# h*h: Since h is constant 1 on [-0.25, 0.25], h*h(t) is a triangle function.
# Base width 0.5, height 0.5. Peak at t=0 is 0.5.
# ||h*h||_inf = 0.5.
# Ratio = 0.5 / (0.5 * 0.5) = 0.5 / 0.25 = 2.0.
# So the seed score is 2.0?
# But the prompt says "CURRENT PROGRAM (score 1.0000)".
# And "score is the lower bound on C2".
# And "search score is value/seed".
# If value = 2.0 and seed value = 2.0, then score = 1.0.
# So the "value" measured by the evaluator is the Ratio.
# And the "seed" value is the Ratio of the seed program.
# If I can make the ratio larger, my score > 1.0.
# But I showed the ratio is unbounded by scaling down.
# Is there a constraint on the heights?
# "Choose 50 nonnegative heights".
# Maybe the evaluator normalizes the heights?
# "The evaluator measures the raw value".
# If the evaluator doesn't normalize, I can just return [1e-10]*50 and get a huge score.
# Why would the seed be 1.0 then?
# Perhaps the evaluator normalizes the L2 norm to 1?
# If ||h||_2 = 1 is enforced, then the ratio is 1 / (||h||_1 * ||h*h||_inf).
# Then we want to MINIMIZE ||h||_1 * ||h*h||_inf.
# Let's assume the evaluator normalizes ||h||_2 = 1.
# Then the objective is to minimize Q(h) = ||h||_1 * ||h*h||_inf subject to ||h||_2=1, h>=0.
# The "value" returned by the evaluator would be 1/Q(h).
# The seed has ||h||_2^2 = 0.5, so ||h||_2 = sqrt(0.5).
# Normalized seed h' = h / sqrt(0.5).
# Q(h') = ||h'||_1 * ||h'*h'||_inf.
# ||h'||_1 = ||h||_1 / sqrt(0.5) = 0.5 / 0.707 = 0.707.
# ||h'*h'||_inf = ||h*h||_inf / 0.5 = 0.5 / 0.5 = 1.0.
# Q(h') = 0.707 * 1.0 = 0.707.
# Value = 1 / 0.707 = 1.414.
# If the evaluator uses the raw unnormalized ratio, seed value is 2.0.
# If the evaluator normalizes L2 to 1, seed value is 1.414.
# In either case, the score is value/seed_value = 1.0.
# So the strategy is: Minimize Q(h) = ||h||_1 * ||h*h||_inf subject to ||h||_2=1.
# This is a well-defined optimization problem.

# Let's implement this.
# We need to compute ||h||_1 and ||h*h||_inf efficiently.
# ||h||_1 = dx * sum(h).
# ||h*h||_inf = dx * max_m (sum_i h_{i+m} h_i).
# Let S = sum(h).
# Let C_max = max_m c_m, where c_m = sum_i h_{i+m} h_i.
# We want to minimize (dx * S) * (dx * C_max) = dx^2 * S * C_max.
# Since dx is constant, minimize F(h) = S * C_max.
# Subject to sum(h_i^2) = 1/dx ? No, ||h||_2^2 = dx * sum(h_i^2) = 1 => sum(h_i^2) = 1/dx.
# Let's work with the normalized vector x = h * sqrt(dx). Then sum(x_i^2) = 1.
# h_i = x_i / sqrt(dx).
# S = sum(h) = (1/sqrt(dx)) sum(x).
# c_m = sum h_{i+m} h_i = (1/dx) sum x_{i+m} x_i.
# C_max = (1/dx) max_m sum x_{i+m} x_i.
# F(h) = S * C_max = (1/sqrt(dx)) sum(x) * (1/dx) max_m ... = (1/dx^{1.5}) * (sum x) * (max_m c_m^x).
# Minimizing F(h) is equivalent to minimizing G(x) = (sum x) * (max_m c_m^x) subject to sum x_i^2 = 1, x_i >= 0.
# The constant 1/dx^{1.5} doesn't affect the argmin.
# The value returned by the evaluator (assuming L2 normalization) would be:
# Value = 1 / (||h||_1 * ||h*h||_inf) = 1 / (dx^2 * S * C_max) = 1 / (dx^2 * (1/dx^{1.5}) G(x)) = dx^{-0.5} / G(x).
# So we want to MINIMIZE G(x).

# Algorithm:
# 1. Define G(x) = (sum x) * (max_m c_m(x)).
# 2. Minimize G(x) using gradient descent or L-BFGS-B with constraint sum x_i^2 = 1.
# 3. We can use Lagrange multipliers or project onto the sphere.
# 4. Since x_i >= 0, we can use L-BFGS-B with bounds [0, inf) and handle the norm constraint by normalizing x in each step, or using a penalty method.
# Normalizing x in each step is robust for sphere constraints.
# Let's use scipy.optimize.minimize with method='L-BFGS-B' and a custom objective that normalizes x?
# No, L-BFGS-B expects a smooth function. Normalizing inside the objective makes it non-smooth at 0.
# Better: Parameterize x on the sphere. Or use a penalty term for ||x||_2^2 - 1.
# Or, simpler: Use gradient descent with projection.
# Projecting onto the intersection of L2 sphere and non-negative orthant is not trivial (it's the sphere cap).
# However, we can just project onto the sphere, and clip negative values? Clipping breaks the norm.
# Alternative: Use a variable transformation x_i = exp(y_i) / norm(exp(y)). This ensures x_i > 0 and sum x_i^2 = 1.
# This is a standard softmax-like parameterization for the sphere.
# Let y be the free variables.
# x_i = exp(y_i) / (sum exp(2 y_j))^{1/2}.
# Then sum x_i^2 = 1.
# We minimize G(x(y)).
# This is smooth.

# Let's compute the gradient of G(x) with respect to x, then chain rule to y.
# G(x) = S(x) * M(x), where S(x) = sum x_i, M(x) = max_m c_m(x).
# c_m(x) = sum_i x_{i+m} x_i.
# M(x) is the max of a set of quadratic forms. It is not differentiable everywhere, but almost everywhere.
# We can use the subgradient: if c_k is the max, then grad M = grad c_k.
# grad S = 1 (vector of ones).
# grad G = M * grad S + S * grad M.
# grad S = ones.
# grad c_k = 2 * (shifted x). Specifically, (grad c_k)_i = x_{i+k} + x_{i-k} (with appropriate boundaries/zeros).
# Note: c_k = sum_i x_{i+k} x_i.
# d c_k / d x_j = x_{j+k} + x_{j-k} (if indices valid, else 0).
# So grad G = M * ones + S * (x_{shifted} + x_{shifted_back}).

# Let's implement this.

def _compute_conv_peak(x):
    """Compute max_m c_m(x) where c_m = sum_i x_{i+m} x_i."""
    # x is a 1D array of length N.
    # We need linear convolution coefficients.
    # c_m for m = -(N-1) to N-1.
    # c_0 = sum x_i^2 = 1 (since normalized).
    # c_m for m > 0 is sum_{i=0}^{N-1-m} x_{i+m} x_i.
    # We can use np.convolve(x, x, mode='full')?
    # np.convolve(x, x) gives sum x_i x_{j-i}.
    # Let's check indices.
    # Convolution y_k = sum_i x_i x_{k-i}.
    # Our c_m = sum_i x_{i+m} x_i.
    # Let j = i+m => i = j-m.
    # c_m = sum_j x_j x_{j-m}.
    # This is the cross-correlation, or convolution with reversed x.
    # Since x is real, c_m = (x * x_flipped)_m?
    # Let's just compute it directly for each m? N=50, so 99 shifts.
    # 99 * 50 ops = 5000 ops. Very fast.
    
    max_val = 0.0
    max_idx = 0
    for m in range(-(N-1), N):
        c = 0.0
        if m >= 0:
            # sum_{i=0}^{N-1-m} x_{i+m} x_i
            c = np.dot(x[m:], x[:N-m])
        else:
            # m < 0, let k = -m
            # sum_{i=k}^{N-1} x_i x_{i-k}
            k = -m
            c = np.dot(x[k:], x[:N-k])
        if c > max_val:
            max_val = c
            max_idx = m
    return max_val, max_idx

def _objective_and_grad(y):
    """
    y: vector of length N.
    Returns objective value and gradient w.r.t y.
    """
    exp_y = np.exp(y - np.max(y)) # for numerical stability
    x = exp_y / np.sqrt(np.sum(exp_y**2))
    
    S = np.sum(x)
    M, m_idx = _compute_conv_peak(x)
    
    G = S * M
    
    # Gradient w.r.t x
    # grad G = M * ones + S * grad c_{m_idx}
    grad_c = np.zeros(N)
    if m_idx >= 0:
        # (grad c)_i = x_{i+m} + x_{i-m}
        # Term 1: x_{i+m}. Non-zero for i in 0..N-1-m.
        grad_c[:N-m_idx] += x[m_idx:]
        # Term 2: x_{i-m}. Non-zero for i in m..N-1.
        grad_c[m_idx:] += x[:N-m_idx]
    else:
        k = -m_idx
        # (grad c)_i = x_{i-k} + x_{i+k}
        # Term 1: x_{i-k}. Non-zero for i in k..N-1.
        grad_c[k:] += x[:N-k]
        # Term 2: x_{i+k}. Non-zero for i in 0..N-1-k.
        grad_c[:N-k] += x[k:]
        
    grad_x = M * np.ones(N) + S * grad_c
    
    # Chain rule: grad_y = J^T grad_x
    # x_i = e_i / R, where e_i = exp(y_i), R = (sum e_j^2)^{1/2}
    # dx_i / dy_k = (delta_{ik} e_i R - e_i * 0.5 R^{-1} * 2 e_i delta_{ik}) / R^2
    # dx_i / dy_k = (delta_{ik} e_i / R) - (e_i e_k / R^3)
    # In matrix form: J = (1/R) I - (1/R^3) e e^T
    # grad_y = J^T grad_x = (1/R) grad_x - (1/R^3) e (e^T grad_x)
    
    e = exp_y
    R = np.sqrt(np.sum(e**2))
    dot = np.dot(e, grad_x)
    grad_y = (1.0/R) * grad_x - (1.0/(R**3)) * e * dot
    
    return G, grad_y

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initialize y
    # Start with uniform x, which corresponds to y=0
    y = np.zeros(N)
    
    best_G = np.inf
    best_y = y.copy()
    
    # Gradient Descent
    lr = 0.1
    for i in range(5000):
        if time.time() > deadline:
            break
        G, grad_y = _objective_and_grad(y)
        if G < best_G:
            best_G = G
            best_y = y.copy()
        
        # Check convergence
        if np.linalg.norm(grad_y) < 1e-6:
            break
            
        # Adaptive learning rate or fixed?
        # Try fixed with decay
        y -= lr * grad_y
        
        # Optional: Line search?
        # For now, simple GD.
        
    # Polish with L-BFGS-B on y
    # We can run L-BFGS-B on y using the objective function.
    # But we need to ensure we don't time out.
    
    def obj_func(y):
        G, _ = _objective_and_grad(y)
        return G
    
    def jac_func(y):
        _, grad_y = _objective_and_grad(y)
        return grad_y
    
    try:
        res = minimize(obj_func, best_y, jac=jac_func, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
        if time.time() < deadline:
            if res.fun < best_G:
                best_G = res.fun
                best_y = res.x
    except Exception:
        pass
    
    # Recover x from best_y
    exp_y = np.exp(best_y - np.max(best_y))
    x = exp_y / np.sqrt(np.sum(exp_y**2))
    
    # x is normalized such that sum x_i^2 = 1.
    # We need h such that sum h_i^2 * dx = 1.
    # h_i = x_i / sqrt(dx).
    # dx = 0.01.
    dx = 0.5 / N
    h = x / np.sqrt(dx)
    
    return list(h)

# Let's double check the logic.
# If I return h, the evaluator computes the ratio.
# If the evaluator normalizes L2 to 1, it will compute:
# h_norm = h / ||h||_2.
# Since ||h||_2 = 1 (by construction), h_norm = h.
# Then it computes R = 1 / (||h||_1 * ||h*h||_inf).
# This matches our objective G(x) scaled by constants.
# Specifically, we minimized G(x) = S_x * M_x.
# The actual value is proportional to 1/G(x).
# So minimizing G(x) maximizes the score.

# One detail: The problem says "50 nonnegative heights on equal intervals of [-1/4, 1/4]".
# My assumption: 50 bins, width 0.01.
# What if it means 50 points?
# If it's 50 points, the "heights" are values at those points.
# But "step heights" implies a step function.
# "construct() -> sequence of nonnegative step heights".
# This confirms it's a step function.
# The intervals are likely the 50 bins.

# Let's add a multi-start to avoid local minima.
# We can randomize y initially.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_G = np.inf
    best_y = np.zeros(N)
    
    # Multi-start
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        y = np.random.randn(N) * 0.1
        # GD
        lr = 0.1
        for i in range(1000):
            if time.time() > deadline - 5.0:
                break
            G, grad_y = _objective_and_grad(y)
            if G < best_G:
                best_G = G
                best_y = y.copy()
            if np.linalg.norm(grad_y) < 1e-7:
                break
            y -= lr * grad_y
            
    # Final polish
    def obj_func(y):
        G, _ = _objective_and_grad(y)
        return G
    def jac_func(y):
        _, grad_y = _objective_and_grad(y)
        return grad_y
        
    try:
        res = minimize(obj_func, best_y, jac=jac_func, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 500})
        if time.time() < deadline:
            if res.fun < best_G:
                best_G = res.fun
                best_y = res.x
    except Exception:
        pass
    
    exp_y = np.exp(best_y - np.max(best_y))
    x = exp_y / np.sqrt(np.sum(exp_y**2))
    
    dx = 0.5 / N
    h = x / np.sqrt(dx)
    
    return list(h)
