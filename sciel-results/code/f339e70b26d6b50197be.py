import numpy as np
import math
import random

N = 150

def construct():
    # Function to compute energy
    def energy(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0)
        d2 = 2.0 - 2.0 * dots
        # Clip to avoid log(0)
        d2 = np.maximum(d2, 1e-15)
        # Sum over i < j
        # E = -0.5 * sum_{i != j} log(sqrt(d2_ij)) = -0.25 * sum_{i != j} log(d2_ij)
        # Using full sum is 2x faster than extracting upper triangle
        # sum_{i != j} log(d2_ij)
        s = np.sum(np.log(d2))
        return -0.25 * s

    # Function to compute gradient
    def grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0)
        d2 = 2.0 - 2.0 * dots
        d2_safe = np.where(d2 > 1e-15, d2, 1e-15)
        
        # diff[i, j] = P_j - P_i
        diff = P[None, :, :] - P[:, None, :]
        
        # terms[i, j] = (P_j - P_i) / d2_ij
        terms = diff / d2_safe[:, :, None]
        
        # G_i = sum_j terms[i, j]
        G = np.sum(terms, axis=1)
        
        # Project to tangent space
        dots_G_P = np.sum(G * P, axis=1)
        G_proj = G - dots_G_P[:, None] * P
        return G_proj

    # Riemannian Exponential Map
    def exp_map(P, v):
        v_norm = np.linalg.norm(v, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_term = np.cos(v_norm_safe)
        sin_term = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_term[:, None] * P + sin_term[:, None] * v
        # Normalize to stay on sphere
        P_new /= np.linalg.norm(P_new, axis=1)[:, None]
        return P_new

    # Riemannian Line Search
    def line_search(P, G, eta0):
        E0 = energy(P)
        eta = eta0
        for _ in range(20):
            v = -eta * G
            P_new = exp_map(P, v)
            E_new = energy(P_new)
            if E_new < E0:
                return P_new, eta
            eta *= 0.5
        # If no step found, return current P and small eta
        return P, eta

    best_P = None
    best_E = float('inf')
    
    # Multiple restarts
    # Use random restarts and a few structured ones
    num_restarts = 10
    
    for r in range(num_restarts):
        if r == 0:
            # Start with Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            rad = np.sqrt(1.0 - z * z)
            P = np.column_stack([rad * np.cos(phi), rad * np.sin(phi), z])
        else:
            # Random initialization
            P = np.random.randn(N, 3)
            P /= np.linalg.norm(P, axis=1)[:, None]
            
        E_curr = energy(P)
        if E_curr < best_E:
            best_E = E_curr
            best_P = P.copy()
            
        eta = 1e-3
        max_iters = 5000
        
        for it in range(max_iters):
            G = grad(P)
            # Check convergence: norm of gradient
            g_norm = np.linalg.norm(G)
            if g_norm < 1e-8:
                break
            
            P_new, eta_used = line_search(P, G, eta)
            
            if P_new is P:
                # No improvement, try smaller eta or break
                eta *= 0.5
                if eta < 1e-10:
                    break
                continue
                
            E_new = energy(P_new)
            
            if E_new < best_E:
                best_E = E_new
                best_P = P_new.copy()
                
            P = P_new
            # Keep eta or increase slightly if it worked well?
            # Simple strategy: keep eta if it worked, reduce if it had to backtrack a lot?
            # For now, just keep the eta that worked.
            
    return best_P
