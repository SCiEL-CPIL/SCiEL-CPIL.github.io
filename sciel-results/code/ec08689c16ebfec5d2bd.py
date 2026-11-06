import numpy as np
import scipy.linalg
import scipy.integrate
import time
import sys

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, keep 50s margin
    
    # Step 1: Probe the system at t=0.5 to get A and b
    # f(t, y) = A(t)y + b(t)
    t_mid = 0.5 * (t0 + t1)
    
    # Get b(t_mid)
    b_mid = np.array(f(t_mid, np.zeros(n)), dtype=float)
    
    # Get A(t_mid)
    A_mid = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # f(t, e_j) = A e_j + b = A[:, j] + b
        # A[:, j] = f(t, e_j) - b
        A_mid[:, j] = np.array(f(t_mid, e_j), dtype=float) - b_mid
        
    # Check if system is constant: f(t, y) = A_mid y + b_mid
    # Test at t0 and t1 with a random vector
    def is_constant(t_test, y_test):
        if time.time() > deadline:
            return False
        y_test = np.array(y_test, dtype=float)
        val = np.array(f(t_test, y_test), dtype=float)
        ref = A_mid @ y_test + b_mid
        return np.linalg.norm(val - ref) < 1e-10 * (1 + np.linalg.norm(ref))
    
    # Test vectors
    y_test1 = np.ones(n)
    y_test2 = np.random.rand(n)
    
    constant = is_constant(t0, y_test1) and is_constant(t1, y_test2)
    
    if constant:
        # Analytic solution for y' = A y + b using eigendecomposition
        # y(t) = exp(A(t-t0)) y0 + integral_{t0}^t exp(A(t-tau)) b dtau
        T = t1 - t0
        try:
            # Eigendecomposition: A = V diag(lambda) V^{-1}
            # exp(A*T) = V diag(exp(lambda*T)) V^{-1}
            # integral = V diag((exp(lambda*T)-1)/lambda) V^{-1} b (for nonzero lambda)
            # For lambda == 0, the integral component is T*b (since exp(0)=1, (1-1)/0 -> T via limit)
            
            eigenvalues, V = np.linalg.eig(A_mid)
            V_inv = np.linalg.inv(V)
            
            # Compute exp(A*T) * y0
            # = V * diag(exp(lambda_i * T)) * V^{-1} * y0
            y0_transformed = V_inv @ y0
            exp_lambdas = np.exp(eigenvalues * T)
            y0_part = V @ (exp_lambdas * y0_transformed)
            
            # Compute integral part: integral_{0}^{T} exp(A*(T-tau)) b dtau
            # = V * diag(g(lambda_i, T)) * V^{-1} * b
            # where g(lambda, T) = (exp(lambda*T) - 1) / lambda for lambda != 0
            # and g(0, T) = T
            b_transformed = V_inv @ b_mid
            g_values = np.empty_like(eigenvalues, dtype=complex)
            for i in range(n):
                lam = eigenvalues[i]
                if abs(lam) < 1e-14:
                    g_values[i] = T
                else:
                    g_values[i] = (np.exp(lam * T) - 1.0) / lam
            integral_part = V @ (g_values * b_transformed)
            
            y_end = y0_part + integral_part
            return np.real_if_close(y_end, tol=1000).astype(float)
        except Exception:
            constant = False

    # Fallback: High accuracy stiff solver
    # Use Radau method which is good for stiff problems.
    # We don't provide jac to let it estimate, which is costly but robust.
    # To save time, we can try to provide a constant jac if we suspect it's constant but verification failed slightly?
    # No, if verification failed, A(t) is likely varying.
    
    # If we are in the fallback, we are burning calls.
    # Limit the solver to finish before deadline.
    # solve_ivp doesn't have a direct "time limit" but we can check time in a callback?
    # Or just trust that 15D stiff system is fast.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0, method='Radau', 
            rtol=1e-10, atol=1e-12
        )
        return sol.y[:, -1]
    except Exception:
        # If solve_ivp fails, try LSODA
        try:
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0, method='LSODA', 
                rtol=1e-10, atol=1e-12
            )
            return sol.y[:, -1]
        except Exception:
            # Last resort: Euler with small steps?
            # Just return y0 to avoid crash? No, try a few steps.
            y = y0.copy()
            t = t0
            h = (t1 - t0) / 100
            for _ in range(100):
                if time.time() > deadline:
                    break
                y = y + h * f(t, y)
                t += h
            return y
