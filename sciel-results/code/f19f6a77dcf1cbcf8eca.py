import numpy as np
from scipy.optimize import linprog
from scipy.special import eval_hermite

def construct():
    """
    Minimize the upper bound C4 on A(f)*A(F(f)) for a Hermite-Gaussian test function.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0.
    
    Returns array of three coefficients [a0, a4, a8].
    """
    
    # Grid setup
    N = 2001
    x = np.linspace(-10, 10, N)
    xi = np.linspace(-10, 10, N)
    dx = x[1] - x[0]
    
    # Weights for trapezoidal rule
    w = np.ones(N) * dx
    w[0] = dx / 2
    w[-1] = dx / 2
    
    # Precompute Hermite polynomials and Gaussians
    H0_x = eval_hermite(0, x)
    H4_x = eval_hermite(4, x)
    H8_x = eval_hermite(8, x)
    H12_x = eval_hermite(12, x)
    
    H0_xi = eval_hermite(0, xi)
    H4_xi = eval_hermite(4, xi)
    H8_xi = eval_hermite(8, xi)
    H12_xi = eval_hermite(12, xi)
    
    exp_x = np.exp(-x**2 / 2)
    exp_xi = np.exp(-xi**2 / 2)
    
    # Values at 0 for a12 calculation
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    # Let coeffs = [a0, a4, a8]
    # g(x) = [a0*(H0 - H0_0/H12_0*H12) + a4*(H4 - H4_0/H12_0*H12) + a8*(H8 - H8_0/H12_0*H12)] * exp(-x^2/2)
    
    # Coefficients for the linear combination in x
    c0_x = (H0_x - (H0_0 / H12_0) * H12_x) * exp_x
    c4_x = (H4_x - (H4_0 / H12_0) * H12_x) * exp_x
    c8_x = (H8_x - (H8_0 / H12_0) * H12_x) * exp_x
    
    # Fourier transform
    # F(Hn e^{-x^2/2}) = (i)^n Hn(xi) e^{-xi^2/2}
    # (i)^0 = 1, (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
    # g_hat(xi) = (a0*1*H0 + a4*1*H4 + a8*1*H8 + a12*(-1)*H12) * exp(-xi^2/2)
    # a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    # g_hat(xi) = [a0*(H0 - H0_0/H12_0*(-1)*H12) + ... ] ?
    # Let's substitute a12:
    # term = a0*H0 + a4*H4 + a8*H8 - (a0*H0_0 + a4*H4_0 + a8*H8_0)/H12_0 * H12
    # Wait, the Fourier of H12 e^{-x^2/2} is (-1) H12(xi) e^{-xi^2/2}.
    # So g_hat = (a0 H0 + a4 H4 + a8 H8 + a12 (-1) H12) e^{-xi^2/2}
    # Substitute a12:
    # g_hat = [ a0 H0 + a4 H4 + a8 H8 - (a0 H0_0 + a4 H4_0 + a8 H8_0)/H12_0 H12 ] e^{-xi^2/2}
    # g_hat = [ a0 (H0 - (H0_0/H12_0) H12) + a4 (H4 - (H4_0/H12_0) H12) + a8 (H8 - (H8_0/H12_0) H12) ] e^{-xi^2/2}
    
    c0_xi = (H0_xi - (H0_0 / H12_0) * H12_xi) * exp_xi
    c4_xi = (H4_xi - (H4_0 / H12_0) * H12_xi) * exp_xi
    c8_xi = (H8_xi - (H8_0 / H12_0) * H12_xi) * exp_xi

    # LP Formulation
    # Variables: a0, a4, a8, u_x (N), v_x (N), u_xi (N), v_xi (N)
    # Total vars: 3 + 4*N
    
    # Objective: min sum w_x (u_x + v_x) + sum w_xi (u_xi + v_xi)
    c_obj = np.zeros(3 + 4 * N)
    c_obj[3:3+N] = w
    c_obj[3+N:3+2*N] = w
    c_obj[3+2*N:3+3*N] = w
    c_obj[3+3*N:3+4*N] = w
    
    # Constraints:
    # 1. g(x_k) - u_x_k <= 0  =>  sum a_i c_i_x_k - u_x_k <= 0
    # 2. -g(x_k) - v_x_k <= 0 => -sum a_i c_i_x_k - v_x_k <= 0
    # 3. g_hat(xi_k) - u_xi_k <= 0
    # 4. -g_hat(xi_k) - v_xi_k <= 0
    
    A_ub = np.zeros((4 * N, 3 + 4 * N))
    
    # For x grid
    # Rows 0 to N-1: g(x) - u_x <= 0
    A_ub[0:N, 0] = c0_x
    A_ub[0:N, 1] = c4_x
    A_ub[0:N, 2] = c8_x
    # u_x is at index 3 + k
    for k in range(N):
        A_ub[k, 3 + k] = -1
        
    # Rows N to 2N-1: -g(x) - v_x <= 0
    A_ub[N:2*N, 0] = -c0_x
    A_ub[N:2*N, 1] = -c4_x
    A_ub[N:2*N, 2] = -c8_x
    # v_x is at index 3 + N + k
    for k in range(N):
        A_ub[N + k, 3 + N + k] = -1
        
    # For xi grid
    # Rows 2N to 3N-1: g_hat(xi) - u_xi <= 0
    A_ub[2*N:3*N, 0] = c0_xi
    A_ub[2*N:3*N, 1] = c4_xi
    A_ub[2*N:3*N, 2] = c8_xi
    # u_xi is at index 3 + 2N + k
    for k in range(N):
        A_ub[2*N + k, 3 + 2*N + k] = -1
        
    # Rows 3N to 4N-1: -g_hat(xi) - v_xi <= 0
    A_ub[3*N:4*N, 0] = -c0_xi
    A_ub[3*N:4*N, 1] = -c4_xi
    A_ub[3*N:4*N, 2] = -c8_xi
    # v_xi is at index 3 + 3N + k
    for k in range(N):
        A_ub[3*N + k, 3 + 3*N + k] = -1
        
    b_ub = np.zeros(4 * N)
    
    # Bounds for a0, a4, a8
    # u, v >= 0
    bounds = [
        (-5, 5),  # a0
        (-1, 1),  # a4
        (-0.01, 0.01), # a8
    ]
    for _ in range(4 * N):
        bounds.append((0, None))
        
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x[:3]
    else:
        # Fallback to seed values
        return np.array([-113/100, 1/25, 1/3240])

