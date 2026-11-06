import numpy as np
import time

def _build_matrices():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain."""
    N = 6
    dim = 2 ** N
    delta = 0.8
    field = 0.3
    
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_chain(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    for (s1, s2) in [(0,1), (2,3), (4,5)]:
        for op1, op2, coeff in [(sx, sx, 1.0), (sy, sy, 1.0), (sz, sz, delta)]:
            ops = []
            for i in range(N):
                if i == s1:
                    ops.append(op1)
                elif i == s2:
                    ops.append(op2)
                else:
                    ops.append(I2)
            A += coeff * kron_chain(ops)
    
    # Even bonds: (1,2), (3,4)
    for (s1, s2) in [(1,2), (3,4)]:
        for op1, op2, coeff in [(sx, sx, 1.0), (sy, sy, 1.0), (sz, sz, delta)]:
            ops = []
            for i in range(N):
                if i == s1:
                    ops.append(op1)
                elif i == s2:
                    ops.append(op2)
                else:
                    ops.append(I2)
            B += coeff * kron_chain(ops)
    
    # Field split
    for i in range(N):
        ops = []
        for j in range(N):
            if j == i:
                ops.append(sz)
            else:
                ops.append(I2)
        field_term = -field * kron_chain(ops)
        A += 0.5 * field_term
        B += 0.5 * field_term
    
    return A, B

def coefficients(stages):
    """Return 4th-order Suzuki-Trotter coefficients for the product formula."""
    # 4th-order coefficients
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    c2 = -2.0**(1.0/3.0) / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    c3 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    
    # 6-stage 4th-order formula
    a6 = [c1, c2, c3, c3, c2, c1]
    b6 = [0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
    
    # Expand to 12 stages by repeating
    a = a6 * 2
    b = b6 * 2
    
    # Normalize to ensure sum = 1
    s_a = sum(a)
    s_b = sum(b)
    if abs(s_a) > 1e-15:
        a = [x / s_a for x in a]
    if abs(s_b) > 1e-15:
        b = [x / s_b for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Final normalization after clamping
    s_a = sum(a)
    s_b = sum(b)
    if abs(s_a) > 1e-15:
        a = [x / s_a for x in a]
    if abs(s_b) > 1e-15:
        b = [x / s_b for x in b]
    
    return a, b
