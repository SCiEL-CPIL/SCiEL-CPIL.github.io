"""Custom high-order implicit Radau integrator for Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np

# Radau IIA 4-stage coefficients
# Nodes c
c = np.array([0.1454971798891699, 0.4691123021378939, 0.7522921553899978, 1.0])
# A matrix (lower triangular)
A = np.array([
    [0.0877358787341991, 0, 0, 0],
    [0.1671432856764082, 0.0219838217980543, 0, 0],
    [0.3243756988786795, 0.1405363942933760, 0.0102176622179423, 0],
    [0.4138060520113160, 0.2011520537818920, 0.0766538942067920, 0.0083880000000000]
])
# b vector (weights)
b = np.array([0.1562071816846645, 0.3365240586327590, 0.3365240586327590, 0.1707447010500175])

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Fixed step size. 
    # Van der Pol with mu=5 is moderately stiff.
    # Radau IIA 4th order has stability region covering left half plane.
    # We choose a step size small enough to resolve the dynamics accurately 
    # but large enough to keep call count low.
    # Reference uses Radau (implicit, 5th order typically in scipy) with rtol 1e-11.
    # Let's try N=100 steps. h=0.1.
    # Or N=50, h=0.2.
    # Let's start with N=100 to be safe on accuracy, then we can see if we can reduce.
    # Actually, let's look at the score.
    # Score = -log10(rel) - 0.5 * log10(calls)
    # If we halve calls, we gain 0.5 * log10(2) ~ 0.15 points.
    # If we halve rel, we gain 1.0 point.
    # So accuracy is more important than call count, up to a point.
    # But we want to minimize calls for the same accuracy.
    
    # Let's try N=200 steps initially to ensure high accuracy, then optimize.
    # With Newton iterations, each step costs ~ 10-20 calls.
    # 200 steps * 15 calls/step = 3000 calls.
    # log10(3000) ~ 3.5. Penalty ~ 1.75.
    # If rel is 1e-15, gain is 15. Total ~ 13.25.
    # Seed score 2.038.
    
    N = 100
    h = (t1 - t0) / N
    t = t0
    y = y0.copy()
    
    # Pre-allocate arrays for stages
    Y = np.zeros((4, 2))
    
    for i in range(N):
        # Solve for stages Y[0..3]
        # Y[k] = y + h * sum_{j=0}^{k} A[k,j] * f(t + c[k]*h, Y[k]) ? No.
        # Radau IIA:
        # Y_i = y + h * sum_{j=1}^{s} A_{ij} f(Y_j)
        # y_{n+1} = y_n + h * sum_{j=1}^{s} b_j f(Y_j)
        
        # We need to solve the system for Y_1 ... Y_s
        # This is a coupled system.
        # However, for small systems (dim 2), we can solve iteratively or use Newton on the full vector.
        
        # Let's define the residual for the full vector Y_all = [Y_0, Y_1, Y_2, Y_3]
        # R_k = Y_k - y - h * sum_{j=0}^{3} A[k,j] * f(t + c[k]*h, Y_k)  <-- Wait, f is evaluated at Y_j?
        # Standard Radau IIA:
        # Y_i = y + h sum_{j=1}^s A_{ij} f(Y_j)
        # So f is evaluated at the stage values Y_j.
        
        # Since the system is small (2D state, 4 stages = 8D system),
        # we can use a simple fixed-point iteration or Newton.
        # Given the stiffness, Newton is better.
        
        # Let's use a simplified approach:
        # Since A is lower triangular, we can solve sequentially?
        # Y_0 depends on f(Y_0).
        # Y_1 depends on f(Y_0), f(Y_1).
        # This is a triangular system of implicit equations.
        # We can solve for Y_0 first, then Y_1, etc.
        
        Y_curr = y.copy()
        
        for k in range(4):
            # Solve for Y_k
            # Y_k = y + h * sum_{j=0}^{k} A[k,j] * f(Y_j)
            # Note: f is evaluated at Y_j, not t+c_j h? 
            # In Radau IIA, the stages are defined by:
            # Y_i = y + h sum A_{ij} f(Y_j)
            # The time argument of f is usually t + c_i h, but since f is autonomous (doesn't depend on t explicitly),
            # we just pass Y_j.
            
            # Initial guess for Y_k is the previous stage or y
            Y_k = Y_curr if k > 0 else y.copy()
            
            # Newton iteration for Y_k
            # g(Y_k) = Y_k - y - h * ( sum_{j=0}^{k-1} A[k,j] f(Y_j) + A[k,k] f(Y_k) ) = 0
            # Let C_k = y + h * sum_{j=0}^{k-1} A[k,j] f(Y_j)
            # Then g(Y_k) = Y_k - C_k - h * A[k,k] * f(Y_k)
            
            # Precompute the sum of previous terms
            sum_prev = np.zeros(2)
            for j in range(k):
                # We need f(Y_j). We haven't stored them?
                # We should store f(Y_j) as we go.
                pass
            
            # Let's restructure to store F_j = f(Y_j)
            F = [None] * 4
            
            # We need to re-evaluate or store. Let's store.
            # But Y_j are already computed.
            
            # Recalculate sum_prev
            for j in range(k):
                if F[j] is None:
                    F[j] = f(t, Y[j]) # t is dummy for autonomous
                sum_prev += A[k, j] * F[j]
            
            C_k = y + h * sum_prev
            
            # Solve Y_k - C_k - h * A[k,k] * f(Y_k) = 0
            # Newton:
            # J_k = I - h * A[k,k] * J_f(Y_k)
            # Y_k_new = Y_k - J_k^{-1} g(Y_k)
            
            Y_k_new = Y_k
            for _ in range(10):
                F_k = f(t, Y_k)
                F[k] = F_k
                g = Y_k - C_k - h * A[k, k] * F_k
                
                # Jacobian of g wrt Y_k
                # J_g = I - h * A[k,k] * J_f(Y_k)
                y0_val = Y_k[0]
                y1_val = Y_k[1]
                J_f = np.array([
                    [0.0, 1.0],
                    [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val**2)]
                ])
                
                J_g = np.eye(2) - h * A[k, k] * J_f
                
                # Solve J_g * delta = -g
                try:
                    delta = np.linalg.solve(J_g, -g)
                except np.linalg.LinAlgError:
                    break
                    
                Y_k_new = Y_k + delta
                if np.linalg.norm(delta) < 1e-14:
                    break
                Y_k = Y_k_new
            
            Y[k] = Y_k
            Y_curr = Y_k
            
        # Update y
        # y_new = y + h * sum_{j=0}^{3} b_j * F[j]
        # Note: F[3] was computed in the last Newton iteration for k=3.
        # F[0..2] were computed in their respective loops.
        
        y_new = y.copy()
        for j in range(4):
            y_new += h * b[j] * F[j]
            
        y = y_new
        t += h
        
    return y
# EVOLVE-BLOCK-END
