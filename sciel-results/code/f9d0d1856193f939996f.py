"""Rewritten program: ODE solver for 15-dim stiff linear system using vector-based scaling-and-squaring."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    if dt == 0:
        return y

    # Estimate the spectral radius or norm of A to determine scaling factor m.
    # We probe A y0 to get a sense of the magnitude.
    # Let v1 = A y0.
    v1 = f(t0, y)
    norm_y = np.linalg.norm(y)
    norm_v1 = np.linalg.norm(v1)
    
    if norm_y == 0:
        return y
        
    # Estimate L ~ ||A||
    L = norm_v1 / norm_y
    
    # We want || (dt / 2^m) A || < 0.25 for fast convergence of Taylor series.
    # dt * L / 2^m < 0.25
    # 2^m > dt * L / 0.25
    # m > log2(dt * L / 0.25)
    if dt * L < 0.25:
        m = 0
    else:
        m = int(np.ceil(np.log2(dt * L / 0.25)))
        
    # Safety cap on m to avoid excessive calls if L is huge?
    # But if L is huge, we need many squarings.
    # 2^m * K calls.
    # If m=20, 10^6 * 10 = 10^7 calls. Might be too many.
    # But for stiff systems, this is necessary for stability/accuracy of Taylor.
    # However, if m is too large, we hit the call limit.
    # Max calls 2,000,000.
    # K=10. 2^m * 10 < 2,000,000 => 2^m < 200,000 => m < 18.
    # If m > 18, we might run out of calls.
    # In that case, we should increase K? No, K is fixed for convergence of small argument.
    # If m is large, dt/2^m is small, so K can be smaller?
    # Actually, if m is large, the argument is very small, so K=5 might suffice.
    # Let's adjust K based on m?
    # For ||W A|| < 0.25, K=10 is safe.
    # If m is capped, error might be high.
    
    # Let's cap m at 15 to keep calls < 327680.
    if m > 15:
        m = 15
        
    W = dt / (2.0 ** m)
    
    # Function to compute exp(W A) v using Taylor series
    def apply_exp_W(v):
        # v is the current vector
        # Compute sum_{k=0}^K (W A)^k / k! v
        # term_0 = v
        # term_k = term_{k-1} * (W / k) * A ? No.
        # term_k = (W^k / k!) A^k v
        # term_{k-1} = (W^{k-1} / (k-1)!) A^{k-1} v
        # term_k = term_{k-1} * (W/k) * A
        # But we need to apply A.
        # A v = f(v).
        # So we need to compute A^k v.
        # Let u_0 = v.
        # u_k = A u_{k-1} = f(u_{k-1}).
        # term_k = (W^k / k!) u_k.
        
        u = v.copy()
        term = u.copy()
        res = u.copy()
        
        K = 10 # Number of Taylor terms
        
        for k in range(1, K + 1):
            u = f(t0, u) # u = A^k v
            term = term * (W / k)
            res = res + term
            
            # Optional: break if term is small
            if np.linalg.norm(term) < 1e-15 * np.linalg.norm(res):
                break
                
        return res

    # Apply exp(W A) 2^m times
    # y_final = (exp(W A))^(2^m) y
    y_curr = y.copy()
    for _ in range(m):
        y_curr = apply_exp_W(y_curr)
        
    return y_curr
# EVOLVE-BLOCK-END
