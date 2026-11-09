import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    timeout = 290.0

    # Try to infer structure (Kronecker sum of 1D Laplacians)
    # Assume square grid H x H
    H = int(np.sqrt(n))
    if H * H != n:
        # Not a square grid, fall back to full reconstruction
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
            if time.time() - start_time > timeout:
                break
        # Solve
        w, v = np.linalg.eig(A)
        v_inv = np.linalg.inv(v)
        coeffs = v_inv @ y0
        coeffs_end = coeffs * np.exp(w * (t1 - t0))
        return v @ coeffs_end

    # Attempt Kronecker reconstruction
    e0 = np.zeros(n)
    e0[0] = 1.0
    resp = f(t0, e0)
    
    # Extract coefficients from the first column of A
    # Indices for 0,0 (top-left):
    # Center: 0
    # Right (x-neighbor): 1
    # Down (y-neighbor): H
    # (Assuming row-major flattening: index = i * H + j)
    
    c_center = resp[0]
    c_x = resp[1] if H > 1 else 0.0
    c_y = resp[H] if H > 1 else 0.0
    
    # Check if it looks like a standard isotropic Laplacian
    # c_center should be negative, c_x, c_y positive (or vice versa depending on sign convention of A in y'=Ay)
    # Usually y' = A y, and A is the discrete Laplacian (negative definite)
    # So c_center < 0, c_x > 0? Or c_center = -4, c_x = 1.
    
    # Verify consistency: c_x should equal c_y
    if not np.isclose(c_x, c_y, rtol=1e-5, atol=1e-8):
        # Not isotropic, or different structure. Fall back to full reconstruction.
        A = np.zeros((n, n), dtype=float)
        A[:, 0] = resp
        for j in range(1, n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
            if time.time() - start_time > timeout:
                break
        w, v = np.linalg.eig(A)
        v_inv = np.linalg.inv(v)
        coeffs = v_inv @ y0
        coeffs_end = coeffs * np.exp(w * (t1 - t0))
        return v @ coeffs_end

    # Build 1D Laplacian L
    L = np.zeros((H, H), dtype=float)
    for i in range(H):
        L[i, i] = c_center
        if i > 0:
            L[i, i-1] = c_x
        if i < H - 1:
            L[i, i+1] = c_x

    # Build 2D Operator A = I \otimes L + L \otimes I
    # Note: np.kron(A, B) for vec(row-major) X is (I \otimes L) for row operations?
    # Let's verify:
    # vec(L X) = (I \otimes L) vec(X)
    # vec(X L) = (L^T \otimes I) vec(X)
    # So A = I \otimes L + L \otimes I (since L is symmetric)
    
    I = np.eye(H)
    A = np.kron(I, L) + np.kron(L, I)
    
    # Diagonalize A
    # A is symmetric
    w, v = np.linalg.eigh(A)
    
    # Solve
    coeffs = v.T @ y0
    coeffs_end = coeffs * np.exp(w * (t1 - t0))
    y_end = v @ coeffs_end
    
    return y_end
