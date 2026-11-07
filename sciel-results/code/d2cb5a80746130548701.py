"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.special import roots_hermite
import sys

def _compute_value(c0, c1, c2):
    """Compute the objective value for given coefficients."""
    try:
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1344.0
        H12_0 = 209088.0
        
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # q(x) = g(x) / x^2
        # Coeffs of q from x^10 to x^0
        # c3 is leading coeff of g, so leading coeff of q is c3 (x^10)
        # We need q(y) where y=x^2. 
        # g(x) = c0 + c1(x^4-6x^2+3) + c2(x^8-28x^6+280x^4-840x^2+105) + c3(x^12-84x^10+1680x^8-16800x^6+60480x^4-30240x^2+10395)
        # q(x) = c3 x^10 - 84 c3 x^8 + (1680 c3 + c2) x^6 - (16800 c3 + 28 c2) x^4 + (60480 c3 + 280 c2 + c1) x^2 - (30240 c3 + 840 c2 + 6 c1)
        # Let y = x^2.
        # q(y) = c3 y^5 - 84 c3 y^4 + (1680 c3 + c2) y^3 - (16800 c3 + 28 c2) y^2 + (60480 c3 + 280 c2 + c1) y - (30240 c3 + 840 c2 + 6 c1)
        
        # Check exact division: constant term of q(y) must be 0?
        # g(0)=0 ensures constant term of g is 0.
        # q(x) = g(x)/x^2. For q to be a polynomial, coeff of x^1 in g must be 0 (it is, even/odd) and coeff of x^0 in g must be 0 (it is).
        # So q(x) is a polynomial in x^2, i.e., q(y).
        # The constant term of q(y) is the coeff of x^2 in g.
        # coeff_x2_g = -30240*c3 - 840*c2 - 6*c1
        # This term is the constant term of q(y). It does NOT have to be 0.
        
        # Build q(y) coefficients [y^5, y^4, y^3, y^2, y^1, y^0]
        a5 = c3
        a4 = -84 * c3
        a3 = 1680 * c3 + c2
        a2 = -16800 * c3 - 28 * c2
        a1 = 60480 * c3 + 280 * c2 + c1
        a0 = -30240 * c3 - 840 * c2 - 6 * c1
        
        # If g tends to negative limit at infinity, g is replaced by -g.
        # Leading term of g is c3 x^12.
        # If c3 < 0, g -> -inf. Then -g is used.
        # -g has leading term -c3 x^12 > 0.
        # The value is r^2 / (2 pi).
        # If we flip g, q becomes -q. Roots are the same. Sign changes flip?
        # "r be the largest real root of q at which q changes sign."
        # If q -> -q, sign change at r is preserved (derivative flips sign, but crossing is same).
        # So we can just assume c3 > 0 WLOG for the search, or handle it.
        # If c3 < 0, we effectively solve for -c3, -c2, -c1, -c0.
        # Value is same.
        
        if a5 == 0:
            # Degenerate, lower degree.
            pass
        
        # Find roots of q(y) in y >= 0
        # q(y) = a5 y^5 + a4 y^4 + a3 y^3 + a2 y^2 + a1 y + a0
        coeffs_y = np.array([a5, a4, a3, a2, a1, a0])
        
        # Filter out leading zeros
        while len(coeffs_y) > 1 and abs(coeffs_y[0]) < 1e-12:
            coeffs_y = coeffs_y[1:]
            
        if len(coeffs_y) == 1:
            # Constant polynomial
            if abs(coeffs_y[0]) < 1e-12:
                # q(y) = 0 for all y?
                return float('inf') # Undefined or 0? If g=0, r is undefined.
            return float('inf')
            
        roots_y = np.roots(coeffs_y)
        
        # Filter real positive roots
        real_roots_y = roots_y[np.abs(roots_y.imag) < 1e-9].real
        pos_roots_y = real_roots_y[real_roots_y > 1e-9]
        
        if len(pos_roots_y) == 0:
            # No positive roots. 
            # If q(y) > 0 for all y>0, then g(x) > 0 for all x!=0.
            # Then r is... ? "largest real root... at which q changes sign".
            # If no sign changing root, is r 0? Or undefined?
            # Contract: "r be the largest real root of q at which q changes sign."
            # If none, value is likely 0 or inf?
            # If q(y) has no real roots, it doesn't change sign.
            # However, if q(0) < 0 and q(inf) > 0, there is a root.
            # If q(y) > 0 for all y, no roots.
            # In that case, the "uncertainty" might be 0?
            # Let's assume if no sign-changing root, value is 0.
            # But wait, if g(x) > 0 for all x, the inequality holds with C=0?
            # Let's check seed. Seed has roots.
            # If no roots, return 0.
            return 0.0
            
        # For each positive root, check if it is a sign change root.
        # q(y) changes sign at y0 if multiplicity is odd.
        # We can check by evaluating q(y0 - eps) and q(y0 + eps).
        
        max_r_y = 0
        for ry in pos_roots_y:
            # Check sign change
            eps = 1e-7 * max(1.0, abs(ry))
            y_left = max(0.0, ry - eps)
            y_right = ry + eps
            
            q_left = np.polyval(coeffs_y, y_left)
            q_right = np.polyval(coeffs_y, y_right)
            
            if q_left * q_right < 0:
                # Sign change
                if ry > max_r_y:
                    max_r_y = ry
                    
        if max_r_y == 0:
            return 0.0
            
        r = np.sqrt(max_r_y)
        value = r**2 / (2 * np.pi)
        return value
        
    except Exception:
        return float('inf')

def construct():
    """Construct three coefficients that minimize the objective value."""
    start_time = time.time()
    deadline = start_time + 200
    
    # We will use Bisection on r.
    # For a given r, we check if there exists c such that:
    # 1. q(r^2) = 0
    # 2. q(y) >= 0 for all y in [0, infinity)
    # 3. c3 > 0 (WLOG)
    
    # q(y) = a5 y^5 + a4 y^4 + a3 y^3 + a2 y^2 + a1 y + a0
    # a_i are linear in c0, c1, c2, c3.
    # c3 is determined by c0, c1, c2 to satisfy g(0)=0.
    # So we can express a_i in terms of c0, c1, c2 only.
    
    # Let's precompute the matrix A such that a = A * c
    # c = [c0, c1, c2]
    # c3 = -(c0 + 12 c1 + 1344 c2) / 209088
    
    # a5 = c3
    # a4 = -84 c3
    # a3 = 1680 c3 + c2
    # a2 = -16800 c3 - 28 c2
    # a1 = 60480 c3 + 280 c2 + c1
    # a0 = -30240 c3 - 840 c2 - 6 c1
    
    # Substitute c3:
    # c3 = -k0 c0 - k1 c1 - k2 c2
    # k0 = 1/209088
    # k1 = 12/209088
    # k2 = 1344/209088
    
    k0 = 1.0 / 209088.0
    k1 = 12.0 / 209088.0
    k2 = 1344.0 / 209088.0
    
    # a5 = -k0 c0 - k1 c1 - k2 c2
    # a4 = -84 (-k0 c0 - k1 c1 - k2 c2) = 84 k0 c0 + 84 k1 c1 + 84 k2 c2
    # a3 = 1680 (-k0 c0 - k1 c1 - k2 c2) + c2 = -1680 k0 c0 - 1680 k1 c1 + (1 - 1680 k2) c2
    # a2 = -16800 (-k0 c0 - k1 c1 - k2 c2) - 28 c2 = 16800 k0 c0 + 16800 k1 c1 + (16800 k2 - 28) c2
    # a1 = 60480 (-k0 c0 - k1 c1 - k2 c2) + c1 + 280 c2 = -60480 k0 c0 + (1 - 60480 k1) c1 + (280 - 60480 k2) c2
    # a0 = -30240 (-k0 c0 - k1 c1 - k2 c2) - 6 c1 - 840 c2 = 30240 k0 c0 + (-6 + 30240 k1) c1 + (-840 + 30240 k2) c2
    
    A = np.array([
        [-k0, -k1, -k2],
        [84*k0, 84*k1, 84*k2],
        [-1680*k0, -1680*k1, 1 - 1680*k2],
        [16800*k0, 16800*k1, 16800*k2 - 28],
        [-60480*k0, 1 - 60480*k1, 280 - 60480*k2],
        [30240*k0, -6 + 30240*k1, -840 + 30240*k2]
    ])
    
    # Feasibility Check for a given r:
    # We want to find c = [c0, c1, c2] such that:
    # 1. a(r^2) = 0  => A * c evaluated at y=r^2 is 0.
    #    This is a linear equation: (A * c)^T * [ (r^2)^5, ..., 1 ] = 0
    #    Let v(r^2) = [ (r^2)^5, (r^2)^4, (r^2)^3, (r^2)^2, r^2, 1 ]
    #    Constraint: v(r^2) @ A @ c = 0
    # 2. q(y) >= 0 for all y in grid.
    #    For each y_i in grid: v(y_i) @ A @ c >= 0
    # 3. c3 > 0 => -k0 c0 - k1 c1 - k2 c2 > 0 => -[k0, k1, k2] @ c > 0
    
    def is_feasible(r):
        if time.time() > deadline:
            return False, None
            
        y_target = r * r
        # Vector for the root constraint
        v_root = np.array([y_target**5, y_target**4, y_target**3, y_target**2, y_target, 1.0])
        # Linear constraint: c^T A^T v_root = 0
        # Let b_root = A.T @ v_root. Then b_root @ c = 0.
        b_root = A.T @ v_root
        
        # Grid for positivity
        # We need to cover [0, infinity). 
        # Since leading coeff a5 must be positive (for g->inf), and q(y)~a5 y^5, 
        # for large y, q(y) is positive if a5>0.
        # We should check grid up to some Y_max.
        # If r is small, Y_max can be moderate.
        Y_max = max(10.0, 5 * y_target)
        grid = np.linspace(0, Y_max, 100)
        
        # Constraints: v(y_i) @ A @ c >= 0
        # Let M be matrix where row i is A.T @ v(grid[i])
        # M @ c >= 0
        M = np.array([A.T @ np.array([y**5, y**4, y**3, y**2, y, 1.0]) for y in grid])
        
        # c3 > 0 constraint:
        # c3 = -k0 c0 - k1 c1 - k2 c2 > 0
        # -[k0, k1, k2] @ c > 0
        # Let's add this as a strict inequality, approximated by >= epsilon
        b_c3 = -np.array([k0, k1, k2])
        
        # Combine into LP:
        # Minimize 0
        # Subject to:
        # b_root @ c = 0
        # M @ c >= 0  => -M @ c <= 0
        # b_c3 @ c >= eps => -b_c3 @ c <= -eps
        
        eps = 1e-6
        
        c_obj = np.zeros(3)
        
        # Inequality constraints: A_ub @ c <= b_ub
        A_ub = np.vstack([
            -M,
            -b_c3.reshape(1, 3)
        ])
        b_ub = np.concatenate([
            np.zeros(len(grid)),
            [-eps]
        ])
        
        # Equality constraints: A_eq @ c = b_eq
        A_eq = b_root.reshape(1, 3)
        b_eq = np.array([0.0])
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
            if res.success:
                c_sol = res.x
                # Verify with _compute_value
                val = _compute_value(c_sol[0], c_sol[1], c_sol[2])
                # Check if the root found is actually r
                # The LP ensures q(r^2)=0. But is it the LARGEST sign changing root?
                # We need to ensure no larger sign-changing root exists.
                # This is hard to enforce in LP.
                # However, if we bisect on r, and for a given r we find a c where q(r^2)=0 and q>=0,
                # it's possible that q has a larger root r' > r where it changes sign.
                # In that case, the true value is r'^2/2pi > r^2/2pi.
                # So finding a feasible c for r doesn't guarantee the value is r^2/2pi.
                # It guarantees the value is <= r^2/2pi IF r is the largest.
                # Wait. If there is a larger root, the value is larger.
                # We want to MINIMIZE the value.
                # So we want the SMALLEST r such that there EXISTS a c with largest sign-changing root = r.
                # If we find c for r, but it has larger root r', then c is valid for r' but not r.
                # So we should check the actual value of the found c.
                if val < r**2 / (2 * np.pi) + 1e-5:
                    # This c is a valid candidate with value <= r^2/2pi
                    # Actually, if val < r^2/2pi, it means the largest root is < r.
                    # So r is not the tightest bound.
                    # We want to find the minimum possible value.
                    # So we should return the actual value of the found c.
                    return True, c_sol
                else:
                    # The found c has a larger root.
                    # This means r is too small to be the LARGEST root for this c.
                    # But maybe another c exists for this r?
                    # The LP found a c with q(r^2)=0. If that c has larger roots, it's not good for minimizing.
                    # We should reject it or try to constrain the roots to be < r.
                    # Constraining "all roots < r" is non-convex.
                    # Alternative: Just use the value of the found c.
                    # If we find a c with value V, we update best.
                    # We don't strictly need bisection on r if we can just find good c's.
                    # But the LP provides a way to find c's with a specific root.
                    # Let's just return the c and let the caller check its value.
                    return True, c_sol
            else:
                return False, None
        except Exception:
            return False, None

    best_c = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_value(*best_c)
    
    # Bisection range for r
    # Seed value
    r_seed = np.sqrt(best_val * 2 * np.pi)
    
    r_low = 0.1
    r_high = r_seed * 2
    
    # Iterate
    for i in range(50):
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2
        feasible, c_sol = is_feasible(r_mid)
        
        if feasible and c_sol is not None:
            val = _compute_value(c_sol[0], c_sol[1], c_sol[2])
            if val < best_val:
                best_val = val
                best_c = c_sol.copy()
            # If we found a solution, the true minimal r might be lower.
            # But we must ensure the found c doesn't have a larger root.
            # If val <= r_mid^2 / 2pi, then r_mid is an upper bound on the optimal r.
            # So we can lower r_high.
            if val <= r_mid**2 / (2 * np.pi) + 1e-6:
                r_high = r_mid
            else:
                # The found c has a larger root.
                # This doesn't mean r_mid is infeasible for being the LARGEST root.
                # It just means THIS c is not good.
                # We should probably move r_high up? Or keep it?
                # If we move r_high up, we might find a c with even larger root.
                # We want to find a c with root = r_mid.
                # The LP constraint q(r_mid^2)=0 is necessary.
                # If all solutions to q(r_mid^2)=0 have larger roots, then r_mid is not the largest root for any solution.
                # So we should increase r_low? No, r_low is the lower bound of the search space.
                # If r_mid is not achievable as the LARGEST root, then the optimal r is > r_mid.
                # So r_low = r_mid.
                r_low = r_mid
        else:
            # Infeasible. No c exists with q(r_mid^2)=0 and q>=0.
            # This means r_mid is too small? Or too large?
            # If r is too small, maybe no polynomial can have its largest root at r and be positive.
            # If r is too large, we can always scale? No, coefficients are fixed form.
            # Usually, if r is too small, it's infeasible.
            r_low = r_mid
            
    # Final local polish
    if time.time() < deadline - 10:
        # Try to improve best_c with local search
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            c_try = best_c + np.random.normal(0, 1e-4, 3)
            val = _compute_value(c_try[0], c_try[1], c_try[2])
            if val < best_val:
                best_val = val
                best_c = c_try
                
    return np.array(best_c)

# EVOLVE-BLOCK-END
