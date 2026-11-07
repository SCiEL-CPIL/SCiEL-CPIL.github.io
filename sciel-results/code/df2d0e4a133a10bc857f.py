"""
Program to optimize coefficients c0, c1, c2 for the uncertainty inequality task.
Combines robust coefficient precomputation with a targeted search strategy.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) = sum_{k=0}^{n} c_k x^k
# We generate them using the recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
# Stored as lists [c_0, c_1, ..., c_n] where c_i is coeff of x^i

def generate_hermite_coeffs(n):
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    h_prev = [1.0]  # H_0
    h_curr = [0.0, 2.0] # H_1
    
    for k in range(1, n):
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        h_next = [0.0] * (k + 2)
        
        # 2x H_k part
        for i in range(k + 1):
            h_next[i + 1] += 2.0 * h_curr[i]
        
        # -2k H_{k-1} part
        for i in range(k):
            h_next[i] += -2.0 * k * h_prev[i]
            
        h_prev = h_curr
        h_curr = h_next
        
    return h_curr

H0_coeffs = generate_hermite_coeffs(0)
H4_coeffs = generate_hermite_coeffs(4)
H8_coeffs = generate_hermite_coeffs(8)
H12_coeffs = generate_hermite_coeffs(12)

# Values at x=0
H0_0 = H0_coeffs[0]
H4_0 = H4_coeffs[0]
H8_0 = H8_coeffs[0]
H12_0 = H12_coeffs[0]

def evaluate_value(c0, c1, c2):
    """
    Calculate the value r^2 / (2*pi) for given coefficients.
    Returns the value.
    """
    # Calculate c3 such that g(0) = 0
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) coefficients (lowest to highest)
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for i in range(len(H0_coeffs)):
        g_coeffs[i] += c0 * H0_coeffs[i]
    for i in range(len(H4_coeffs)):
        g_coeffs[i] += c1 * H4_coeffs[i]
    for i in range(len(H8_coeffs)):
        g_coeffs[i] += c2 * H8_coeffs[i]
    for i in range(len(H12_coeffs)):
        g_coeffs[i] += c3 * H12_coeffs[i]
        
    # q(x) = g(x) / x^2
    # g(0) is 0 by construction. g is even, so g'(0)=0.
    # We can divide by x^2 by dropping the first two coefficients (x^0 and x^1 terms).
    # Note: Due to floating point errors, g_coeffs[0] and g_coeffs[1] might be small non-zeros.
    # We assume they are effectively 0.
    q_coeffs = g_coeffs[2:]
    
    # Remove trailing zeros for np.roots to avoid numerical issues
    while len(q_coeffs) > 0 and abs(q_coeffs[-1]) < 1e-12:
        q_coeffs.pop()
        
    if not q_coeffs:
        return 0.0 # Degenerate case
        
    # Find roots
    # np.roots expects highest degree first
    q_roots = np.roots(q_coeffs[::-1])
    
    # Filter real roots
    real_roots = []
    for root in q_roots:
        if abs(root.imag) < 1e-8:
            real_roots.append(root.real)
            
    if not real_roots:
        # If no real roots, q(x) is always positive or always negative.
        # If q(x) > 0, the bound might be considered 0 or invalid? 
        # The task says "r be the largest real root...". If none, value is undefined?
        # However, typically in these problems, we want to push the root to 0 or make it non-existent.
        # If q(x) is strictly positive, the "largest root" doesn't exist.
        # Let's assume a large penalty or 0. 
        # Looking at the seed, it has roots. 
        # If we can make q(x) > 0 for all x, that would be ideal (value 0?).
        # But "lower is better". 
        # Let's check the sign at a large x.
        # If q(x) > 0 at infinity, and no roots, it's positive everywhere.
        # This likely corresponds to a very good (small) value.
        # However, the evaluator might crash or return inf.
        # Let's return a small positive number to encourage this direction.
        q_large = np.polyval(q_coeffs[::-1], 100.0)
        if q_large > 0:
            return 1e-9
        else:
            # If q(x) < 0 at infinity, it's negative everywhere.
            # g is replaced by -g. Then q becomes -q, which is positive everywhere.
            # So this is also a "good" case.
            return 1e-9

    # Find sign-changing roots
    valid_roots = []
    eps = 1e-6
    for r in real_roots:
        # Check sign change
        q_left = np.polyval(q_coeffs[::-1], r - eps)
        q_right = np.polyval(q_coeffs[::-1], r + eps)
        
        if q_left * q_right < 0:
            # r is a sign-changing root.
            # "largest real root". Since q is even, roots come in pairs +/-r.
            # We want the largest value of r. Since r can be negative, we take abs?
            # "largest real root" usually means the maximum value on the real line.
            # If roots are -5 and 5, largest is 5.
            # If roots are -5 and -2, largest is -2? But q is even, so if -5 is root, 5 is root.
            # So we just take the max of the absolute values? Or the max of the values?
            # The value is r^2 / (2*pi). So it depends on |r|.
            # We want to minimize r^2, so we want the smallest |r|?
            # NO. "r be the largest real root".
            # If the roots are 2 and 5, r=5. Value = 25/(2pi).
            # If the roots are 1 and 10, r=10. Value = 100/(2pi).
            # We want to MINIMIZE the value.
            # So we want to MINIMIZE the largest sign-changing root.
            # This means we want to push the outermost roots inwards.
            
            valid_roots.append(abs(r))
            
    if not valid_roots:
        # No sign changing roots.
        # This means q(x) does not change sign.
        # This is the "always positive" or "always negative" case (after potential flip).
        # As discussed, this is likely the best case (value 0 or close to it).
        return 1e-9

    r_max = max(valid_roots)
    
    # Check if g tends to -inf.
    # Leading coeff of g is c3 * H12[12].
    # H12[12] is 4096.
    # If c3 * 4096 < 0, g -> -inf.
    # Evaluator replaces g with -g.
    # This flips sign of q.
    # Roots and sign changes are invariant.
    # So r_max is unchanged.
    
    value = (r_max ** 2) / (2 * np.pi)
    return value

def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Baseline
    base_val = evaluate_value(seed[0], seed[1], seed[2])
    best_c = seed.copy()
    best_val = base_val
    
    start_time = time.time()
    time_limit = 170.0 # Safety margin before 210s kill
    
    # Define objective function for optimizer
    def obj(c):
        val = evaluate_value(c[0], c[1], c[2])
        return val
        
    # Local Optimization from seed
    # The seed is already quite good (score 1.0 implies it's the baseline).
    # We try to improve it slightly.
    try:
        # Nelder-Mead is robust for low-dimensional non-smooth problems
        res = minimize(obj, seed, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_c = res.x
            best_val = res.fun
    except Exception:
        pass
        
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return best_c
        
    # Try a few random restarts if time permits
    # The search space is small, so we can afford a few random starts.
    np.random.seed(42)
    for _ in range(10):
        if time.time() - start_time > time_limit - 20:
            break
            
        # Random perturbation
        # Scale based on seed magnitude
        noise_scale = 0.1
        c_init = seed + np.random.normal(0, noise_scale, 3)
        
        try:
            res = minimize(obj, c_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_c = res.x
                best_val = res.fun
        except Exception:
            pass
            
    return best_c
# EVOLVE-BLOCK-END
