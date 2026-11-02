"""Optimized Fourier-uncertainty inequality: tune Hermite coefficients to minimize C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Physical constants
hbar = 1.0
m = 1.0
k = 1.0
w0 = np.sqrt(k / m)  # = 1.0

def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_prev = 2.0 * x
    h_curr = 4.0 * x**2 - 2.0
    for _ in range(2, n + 1):
        h_prev, h_curr = h_curr, 2.0 * x * h_curr - 2.0 * _ * h_prev
    return h_curr

def _basis_coeffs():
    """Return list of (H_n, coefficient) for n in {0,4,8,12} with g(0)=0 constraint."""
    # We need H_n(0) values
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 1680
    # H_12(0) = -103950
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # => a12 = -(a0*1 + a4*(-12) + a8*1680) / (-103950)
    #        = (a0 - 12*a4 + 1680*a8) / 103950
    return None  # placeholder

def _compute_c4(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(f) * A(F) for the test function
    g(x) = [a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)] * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    The uncertainty product bound involves integrals of g^2, x^2*g^2, 
    and (Fourier transform related quantities).
    
    For the standard Hermite-Gaussian basis, we can compute:
    - <g|x^2|g> / <g|g>  (position variance related)
    - <g|p^2|g> / <g|g>  (momentum variance related)
    
    where p = -i d/dx (in units where hbar=1, m=1)
    
    The product A(f)*A(F) is bounded by a function of these moments.
    For the specific C4 bound in the uncertainty inequality context,
    we need to compute the relevant integrals.
    
    Using the Hermite polynomial orthogonality:
    ∫ H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) δ_mn
    
    But our Gaussian is exp(-x^2/2), so we need to be careful.
    Let's use the standard form where the Hermite functions are
    ψ_n(x) = (1/(sqrt(2^n n! sqrt(pi)))) H_n(x) exp(-x^2/2)
    
    For g(x) = sum a_n H_n(x) exp(-x^2/2), we can write
    g(x) = sum b_n ψ_n(x) where b_n = a_n sqrt(2^n n! sqrt(pi))
    
    Then <g|g> = sum |b_n|^2 = sum a_n^2 2^n n! sqrt(pi)
    
    For x^2: in the Hermite basis, x = (a† + a)/sqrt(2) where a, a† are
    ladder operators. x^2 = (a†^2 + a^2 + a†a + aa†)/2 = (a†^2 + a^2 + 2a†a + 1)/2
    
    So <x^2> in state ψ_n is n + 1/2
    
    For a superposition g = sum c_n ψ_n (normalized), 
    <x^2> = sum_n |c_n|^2 (n + 1/2) + sum_{n≠m} c_n* c_m <ψ_n|x^2|ψ_m>
    
    The off-diagonal terms: <ψ_n|x^2|ψ_m> is non-zero only when |n-m|=2
    <ψ_n|x^2|ψ_{n+2}> = sqrt((n+1)(n+2))/2
    <ψ_n|x^2|ψ_{n-2}> = sqrt(n(n-1))/2
    
    Similarly for p^2 = -d^2/dx^2, <p^2> in ψ_n is also n + 1/2, and the
    off-diagonal structure is the same.
    
    Actually, for the uncertainty product, we need:
    A(f) = sqrt(<x^2> - <x>^2) and A(F) = sqrt(<p^2> - <p>^2)
    
    By symmetry (even function), <x> = 0 and <p> = 0, so:
    A(f)^2 = <x^2> and A(F)^2 = <p^2>
    
    And A(f)*A(F) = sqrt(<x^2> * <p^2>)
    
    For our even function with only even n, <x> = <p> = 0 automatically.
    """
    # Coefficients a_n for n = 0, 4, 8, 12
    a12 = (a0 - 12.0 * a4 + 1680.0 * a8) / 103950.0
    coeffs = {0: a0, 4: a4, 8: a8, 12: a12}
    
    # Normalization: <g|g> = sum a_n^2 * 2^n * n! * sqrt(pi)
    # We'll work with unnormalized and divide
    norm2 = 0.0
    for n, a in coeffs.items():
        norm2 += a**2 * (2.0**n) * np.math.factorial(n) * np.sqrt(np.pi)
    
    if norm2 <= 0:
        return np.inf
    
    # <x^2> = sum_n |c_n|^2 (n+1/2) + off-diagonal terms
    # where c_n = a_n * sqrt(2^n n! sqrt(pi)) / sqrt(norm2)
    # But actually, let's compute <g|x^2|g> / <g|g> directly
    
    # <g|x^2|g> = sum_{n,m} a_n a_m <H_n exp(-x^2/2) | x^2 | H_m exp(-x^2/2)>
    # = sum_{n,m} a_n a_m * sqrt(pi) * 2^{(n+m)/2} * sqrt(n! m!) * <ψ_n|x^2|ψ_m>
    
    # <ψ_n|x^2|ψ_m> = (n+1/2) δ_nm + (1/2)sqrt((n+1)(n+2)) δ_{m,n+2} + (1/2)sqrt(n(n-1)) δ_{m,n-2}
    
    x2_matrix = {}
    for n in coeffs:
        for m in coeffs:
            val = 0.0
            if n == m:
                val += (n + 0.5)
            if m == n + 2:
                val += 0.5 * np.sqrt((n + 1) * (n + 2))
            if m == n - 2:
                val += 0.5 * np.sqrt(n * (n - 1))
            x2_matrix[(n, m)] = val
    
    x2_expect = 0.0
    for n in coeffs:
        for m in coeffs:
            x2_expect += coeffs[n] * coeffs[m] * np.sqrt(2.0**n * np.math.factorial(n)) * np.sqrt(2.0**m * np.math.factorial(m)) * x2_matrix[(n, m)]
    x2_expect *= np.sqrt(np.pi)
    
    # Similarly for p^2: <ψ_n|p^2|ψ_m> has the same structure as x^2
    # because p = -i d/dx and the ladder operator structure is identical
    # <ψ_n|p^2|ψ_m> = (n+1/2) δ_nm + (1/2)sqrt((n+1)(n+2)) δ_{m,n+2} + (1/2)sqrt(n(n-1)) δ_{m,n-2}
    
    p2_expect = x2_expect  # same by symmetry of the harmonic oscillator
    
    # A(f) * A(F) = sqrt(<x^2> * <p^2>) = sqrt(x2_expect/norm2 * p2_expect/norm2)
    x2_norm = x2_expect / norm2
    p2_norm = p2_expect / norm2
    
    if x2_norm < 0 or p2_norm < 0:
        return np.inf
    
    return np.sqrt(x2_norm * p2_norm)

def construct():
    """
    Find coefficients a0, a4, a8 that minimize the upper bound on C4.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin
    
    # Initial guess: the seed values
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Try multiple starting points
    best_x = x0.copy()
    best_val = _compute_c4(*x0)
    
    # Define objective function
    def objective(x):
        return _compute_c4(x[0], x[1], x[2])
    
    # Try SLSQP from multiple initial points
    initial_points = [
        x0,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([-0.5, -0.5, -0.5]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ]
    
    for x_init in initial_points:
        if time.time() > deadline - 10.0:
            break
        try:
            result = opt.minimize(objective, x_init, method='SLSQP', 
                                  options={'maxiter': 1000, 'ftol': 1e-12})
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead from the best SLSQP result
    if time.time() < deadline - 5.0:
        try:
            result = opt.minimize(objective, best_x, method='Nelder-Mead',
                                  options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tight tolerances
    if time.time() < deadline - 2.0:
        try:
            result = opt.minimize(objective, best_x, method='SLSQP',
                                  options={'maxiter': 2000, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
