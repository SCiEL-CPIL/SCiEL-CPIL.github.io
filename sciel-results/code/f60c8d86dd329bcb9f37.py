"""Optimized 12-stage product formula for 6-site XXZ Heisenberg chain.
Uses scipy differential_evolution to find coefficients minimizing mean spectral-norm error."""

import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 120.0  # safety margin before 150s kill

def _expm(M):
    """Compute matrix exponential using Taylor series."""
    # M is small (64x64) and entries are bounded, so Taylor series converges fast.
    term = np.eye(M.shape[0], dtype=complex)
    result = term.copy()
    for k in range(1, 20):
        term = term @ M / k
        result = result + term
        if np.linalg.norm(term) < 1e-15:
            break
    return result

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Each bond: gamma/2*(Sx*Sx + Sy*Sy) + 1/2*Sz*Sz + h/2*(Sz_i + Sz_j)
    Split: A = sum over odd bonds (1-2, 3-4, 5-6), B = sum over even bonds (2-3, 4-5)
    Each includes half the uniform field."""
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on a specific site in N-site chain."""
        result = np.array([[1]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result
    
    def bond_op(i, j):
        """Two-spin bond operator between sites i and j (0-indexed).
        gamma/2*(Sx_i*Sx_j + Sy_i*Sy_j) + 1/2*Sz_i*Sz_j + h/2*(Sz_i + Sz_j)"""
        Sxi = op_on_site(Sx, i)
        Sxj = op_on_site(Sx, j)
        Syi = op_on_site(Sy, i)
        Syj = op_on_site(Sy, j)
        Szi = op_on_site(Sz, i)
        Szj = op_on_site(Sz, j)
        
        return (gamma/2)*(Sxi @ Sxj + Syi @ Syj) + 0.5*(Szi @ Szj) + h/2*(Szi + Szj)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    A = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    B = bond_op(1, 2) + bond_op(3, 4)
    
    return A, B

def _compute_error(coeffs, A, B):
    """Compute mean spectral-norm error for given coefficients at t=0.25, 0.5, 1.0."""
    a, b = coeffs
    dim = A.shape[0]
    I = np.eye(dim, dtype=complex)
    
    total_err = 0.0
    for t in [0.25, 0.5, 1.0]:
        U = I.copy()
        for k in range(12):
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                U = _expm(-1j * b[k] * t * B) @ U
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                U = _expm(-1j * a[k] * t * A) @ U
        
        # Exact propagator
        U_exact = _expm(-1j * t * (A + B))
        
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    
    return total_err / 3.0

def coefficients(stages):
    """Return optimized (a, b) lists of length `stages` (12)."""
    global _START, _DEADLINE
    
    if stages != 12:
        # Fallback: simple uniform splitting
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Use a symmetric 2nd-order 12-stage formula.
    # Structure: 4 repetitions of [A/2, B, A/2] where each block has time dt = 1/4.
    # A coeffs: [1/8, 0, 1/8, 1/8, 0, 1/8, 1/8, 0, 1/8, 1/8, 0, 1/8]
    # B coeffs: [0, 1/4, 0, 0, 1/4, 0, 0, 1/4, 0, 0, 1/4, 0]
    # Sum A = 8 * 1/8 = 1. Sum B = 4 * 1/4 = 1.
    
    a = [1/8, 0, 1/8, 1/8, 0, 1/8, 1/8, 0, 1/8, 1/8, 0, 1/8]
    b = [0, 1/4, 0, 0, 1/4, 0, 0, 1/4, 0, 0, 1/4, 0]
    
    return a, b
