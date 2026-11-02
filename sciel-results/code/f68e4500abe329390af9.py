import numpy as np
import scipy.optimize as so
import time

def construct():
    start_time = time.time()
    deadline = start_time + 200  # Leave 10s buffer for overhead
    
    # Coefficients of H_{2k}(x) as polynomial in y=x^2
    # H_0(y) = 1
    # H_4(y) = y^2 - 6y + 3
    # H_8(y) = y^4 - 28y^3 + 210y^2 - 420y + 105
    # H_12(y) = y^6 - 60y^5 + 1500y^4 - 12600y^3 + 25200y^2 - 15120y + 10395
    
    # We use the basis of monomials in y for easier integration?
    # Or just evaluate the polynomial.
    
    # Let's define the polynomial P(y) = sum_{j=0}^3 c_j H_{2j}(x) ?
    # No, the input is c0, c4, c8 for H0, H4, H8.
    # Let's denote the coefficients as a, b, c.
    # P(y) = a*H0(y) + b*H4(y) + c*H8(y) + d*H12(y)
    
    # H0(y) coeffs: [1] (constant term)
    # H4(y) coeffs: [3, -6, 1] (const, y, y^2)
    # H8(y) coeffs: [105, -420, 210, -28, 1]
    # H12(y) coeffs: [10395, -15120, 25200, -12600, 1500, -60, 1]
    
    # To compute integrals, it's easier to work with the monomial coefficients of P(y).
    # P(y) = p0 + p1*y + p2*y^2 + p3*y^3 + p4*y^4 + p5*y^5 + p6*y^6
    # We only go up to H12, so degree 6 in y.
    
    # Let's map (a, b, c) to monomial coeffs p[0..6]
    # a*H0 + b*H4 + c*H8 + d*H12
    # d is determined by P(0)=0 => p0 = 0.
    # p0 = a*1 + b*3 + c*105 + d*10395 = 0 => d = -(a + 3b + 105c)/10395
    
    # p1 = b*(-6) + c*(-420) + d*(-15120)
    # p2 = b*1 + c*210 + d*25200
    # p3 = c*(-28) + d*(-12600)
    # p4 = c*1 + d*1500
    # p5 = d*(-60)
    # p6 = d*1
    
    # The integrals are of the form \int_0^\infty y^{k-1/2} e^{-y} P(y)^2 dy
    # Let M_{i,j} = \int_0^\infty y^{i+j-1/2} e^{-y} dy = Gamma(i+j+1/2)
    # Wait, |f|^2 = e^{-y} P(y)^2.
    # \int |f|^2 dx = \frac{1}{2} \int_0^\infty y^{-1/2} e^{-y} P(y)^2 dy
    # \int x^2 |f|^2 dx = \frac{1}{2} \int_0^\infty y^{1/2} e^{-y} P(y)^2 dy
    
    # Let I_0 = \int_0^\infty y^{-1/2} e^{-y} P(y)^2 dy
    # Let I_1 = \int_0^\infty y^{1/2} e^{-y} P(y)^2 dy
    # V_x = I_1 / I_0
    
    # Similarly for Q(y) which is the Fourier side polynomial.
    # Q(s) = \sum c_k (1-s)^k ? No.
    # Recall \mathcal{F}[x^2 e^{-x^2/2}] \propto (1-\xi^2) e^{-\xi^2/2}.
    # If P(x^2) = \sum p_k x^{2k}, then \hat{f}(\xi) \propto e^{-\xi^2/2} \sum p_k (1-\xi^2)^k.
    # So Q(s) = \sum_{k=0}^6 p_k (1-s)^k.
    # We need to compute J_0 and J_1 for Q(s) similarly.
    # J_0 = \int_0^\infty s^{-1/2} e^{-s} Q(s)^2 ds
    # J_1 = \int_0^\infty s^{1/2} e^{-s} Q(s)^2 ds
    # V_\xi = J_1 / J_0
    
    # Objective: V_x * V_\xi
    
    # Precompute Gamma values?
    # Gamma(n + 0.5) for n up to 12 (since P^2 has degree 12, y^{12} * y^{-1/2} = y^{11.5}, Gamma(12.5))
    
    def get_monomial_coeffs(a, b, c):
        d = -(a + 3*b + 105*c) / 10395.0
        p = np.zeros(7)
        # H0: [1]
        p[0] += a
        # H4: [3, -6, 1]
        p[0] += 3*b
        p[1] += -6*b
        p[2] += 1*b
        # H8: [105, -420, 210, -28, 1]
        p[0] += 105*c
        p[1] += -420*c
        p[2] += 210*c
        p[3] += -28*c
        p[4] += 1*c
        # H12: [10395, -15120, 25200, -12600, 1500, -60, 1]
        p[0] += 10395*d
        p[1] += -15120*d
        p[2] += 25200*d
        p[3] += -12600*d
        p[4] += 1500*d
        p[5] += -60*d
        p[6] += 1*d
        return p
        
    def get_Q_coeffs(p):
        # Q(s) = sum p_k (1-s)^k
        # Expand (1-s)^k = sum_{j=0}^k binom(k,j) (-1)^j s^j
        # q_j = sum_{k=j}^6 p_k binom(k,j) (-1)^j
        q = np.zeros(7)
        for k in range(7):
            if p[k] == 0: continue
            for j in range(k+1):
                # binom(k, j)
                from math import comb
                q[j] += p[k] * comb(k, j) * ((-1)**j)
        return q

    def compute_variance(p):
        # I_0 = sum_{i,j} p_i p_j Gamma(i+j-0.5)
        # I_1 = sum_{i,j} p_i p_j Gamma(i+j+0.5)
        # p has length 7 (indices 0..6)
        # Max i+j = 12. Gamma(12.5) and Gamma(11.5)
        # We can precompute Gamma values
        pass

    # Precompute Gamma
    from scipy.special import gamma
    gam_0 = np.array([gamma(i + 0.5) for i in range(13)]) # i+j-0.5, min i+j=0 -> Gamma(0.5)
    gam_1 = np.array([gamma(i + 1.5) for i in range(13)]) # i+j+0.5, min i+j=0 -> Gamma(1.5)
    
    def calc_I(p, gam):
        # p is vector of length 7
        # I = sum_{i=0}^6 sum_{j=0}^6 p_i p_j gam[i+j]
        # Use matrix multiplication for speed? Or just loops (7x7=49 ops, very fast)
        I = 0.0
        for i in range(7):
            if p[i] == 0: continue
            for j in range(7):
                I += p[i] * p[j] * gam[i+j]
        return I

    def objective(coeffs):
        a, b, c = coeffs
        p = get_monomial_coeffs(a, b, c)
        I0 = calc_I(p, gam_0)
        I1 = calc_I(p, gam_1)
        if I0 <= 0: return 1e10
        Vx = I1 / I0
        
        q = get_Q_coeffs(p)
        J0 = calc_I(q, gam_0)
        J1 = calc_I(q, gam_1)
        if J0 <= 0: return 1e10
        Vxi = J1 / J0
        
        return Vx * Vxi

    # Optimization
    best_score = 1e10
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    # Try a few random starts and the seed
    starts = [
        [-1.13, 0.04, 0.0003],
        [-1.0, 0.0, 0.0],
        [-1.5, 0.1, -0.01],
        [-0.5, -0.1, 0.05],
        [0.0, 0.0, 0.0]
    ]
    
    for start in starts:
        if time.time() > deadline: break
        try:
            res = so.minimize(objective, start, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    # If we have time, do a finer grid or more iterations
    if time.time() < deadline - 20:
        # Local refinement
        res = so.minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x

    return np.array(best_coeffs)
