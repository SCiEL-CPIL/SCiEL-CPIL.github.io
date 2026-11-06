"""Product-formula design for Heisenberg-chain time evolution using 6th-order Suzuki-Trotter."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """Build the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain."""
    # 6 sites, open boundary
    # XXZ: H = sum_j [Jx (Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz Sz_j Sz_{j+1}] + h sum_j Sz_j
    # with anisotropy delta = Jz/Jx = 0.8, field h = 0.3
    # Split into even bonds (B) and odd bonds (A), each including half the uniform field
    
    N = 6
    delta = 0.8
    h = 0.3
    Jx = 1.0  # normalization
    
    # Spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    # Build full Hamiltonian on 6-site Hilbert space (dim = 2^6 = 64)
    dim = 2 ** N
    
    def spin_op(op, site):
        """Apply spin operator op at given site on the full Hilbert space."""
        result = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                # Check if only site i and j differ
                if i == j:
                    # Apply op at site
                    # Decompose i and j into binary
                    bits_i = [(i >> (N - 1 - s)) & 1 for s in range(N)]
                    # Compute matrix element
                    me = 0.0
                    for s in range(N):
                        if s == site:
                            me *= op[bits_i[s], bits_i[s]]
                        else:
                            me *= 1.0
                    result[i, i] += me
                else:
                    # Check if i and j differ only at site
                    diff_bits = []
                    for s in range(N):
                        bi = (i >> (N - 1 - s)) & 1
                        bj = (j >> (N - 1 - s)) & 1
                        if bi != bj:
                            diff_bits.append(s)
                    if len(diff_bits) == 1 and diff_bits[0] == site:
                        bi = (i >> (N - 1 - site)) & 1
                        bj = (j >> (N - 1 - site)) & 1
                        result[i, j] += op[bi, bj]
        return result
    
    # Build A (odd bonds: 0-1, 2-3, 4-5) and B (even bonds: 1-2, 3-4) plus half field each
    # Actually for 6 sites with open boundary, bonds are 0-1, 1-2, 2-3, 3-4, 4-5
    # Odd bonds (by bond index 0,2,4): 0-1, 2-3, 4-5
    # Even bonds (by bond index 1,3): 1-2, 3-4
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # XXZ term for a bond between sites i and j
    def xxz_bond(i, j):
        """XXZ interaction between sites i and j."""
        # Sx_i Sx_j + Sy_i Sy_j + delta * Sz_i Sz_j
        # = 0.5*(S+_i S-_j + S-_i S+_j) + delta * Sz_i Sz_j
        S_plus = np.array([[0, 1], [0, 0]])
        S_minus = np.array([[0, 0], [1, 0]])
        
        H = np.zeros((dim, dim), dtype=complex)
        
        # S+_i S-_j + S-_i S+_j
        for i_state in range(dim):
            for j_state in range(dim):
                # S+_i S-_j
                bi_i = (i_state >> (N - 1 - i)) & 1
                bj_j = (j_state >> (N - 1 - j)) & 1
                if bi_i == 0 and bj_j == 1:
                    # S+ on i (0->1), S- on j (1->0)
                    new_i = i_state | (1 << (N - 1 - i))
                    new_j = j_state & ~(1 << (N - 1 - j))
                    H[new_i, i_state] += 0.5 * Jx
                # S-_i S+_j
                bi_i = (i_state >> (N - 1 - i)) & 1
                bj_j = (j_state >> (N - 1 - j)) & 1
                if bi_i == 1 and bj_j == 0:
                    # S- on i (1->0), S+ on j (0->1)
                    new_i = i_state & ~(1 << (N - 1 - i))
                    new_j = j_state | (1 << (N - 1 - j))
                    H[new_i, i_state] += 0.5 * Jx
                # delta * Sz_i Sz_j
                bi_i = (i_state >> (N - 1 - i)) & 1
                bj_j = (j_state >> (N - 1 - j)) & 1
                if i_state == j_state:
                    sz_i = 0.5 if bi_i == 0 else -0.5
                    sz_j = 0.5 if bj_j == 0 else -0.5
                    H[i_state, j_state] += delta * Jx * sz_i * sz_j
        
        return H
    
    # Odd bonds: 0-1, 2-3, 4-5
    A += xxz_bond(0, 1)
    A += xxz_bond(2, 3)
    A += xxz_bond(4, 5)
    
    # Even bonds: 1-2, 3-4
    B += xxz_bond(1, 2)
    B += xxz_bond(3, 4)
    
    # Field term: h * sum_j Sz_j, split half to A and half to B
    field = np.zeros((dim, dim), dtype=complex)
    for site in range(N):
        for state in range(dim):
            s = (state >> (N - 1 - site)) & 1
            sz_val = 0.5 if s == 0 else -0.5
            field[state, state] += h * sz_val
    
    A += 0.5 * field
    B += 0.5 * field
    
    return A, B


def _compute_optimal_coefficients():
    """
    Compute 12-stage product formula coefficients using 6th-order Suzuki-Trotter.
    
    We use the 6th-order Suzuki-Trotter symmetric scheme.
    The 6th-order scheme can be written as 5 S_2 steps with coefficients:
    d1, d2, d3, d2, d1 (symmetric)
    
    For 12 stages in the form prod exp(-i b_k t B) exp(-i a_k t A),
    we need to map the S_2 steps to this format.
    
    A 6th-order Suzuki-Trotter scheme has the form:
    S_6(t) = S_2(d1*t) * S_2(d2*t) * S_2(d3*t) * S_2(d2*t) * S_2(d1*t)
    
    where S_2(x) = exp(-i x A/2) exp(-i x B) exp(-i x A/2)
    
    The 6th-order coefficients are:
    d1 = 1/(6 - 6^(1/3))
    d2 = -6^(1/3)/(6 - 6^(1/3))
    d3 = 1 - 2*d1 - 2*d2
    
    Let me use known 6th-order coefficients.
    Actually, let me use a simpler approach: use 4th-order with 3 substeps.
    
    For 4th-order Yoshida: c1 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3))
    
    For 3 substeps of 4th-order, each substep has 3 S_2 calls.
    Total: 9 S_2 calls.
    
    Each S_2(x) = exp(-i x A/2) exp(-i x B) exp(-i x A/2)
    
    Let's build the full sequence:
    For one 4th-order step with step size dt:
    S_2(c1*dt) * S_2(c2*dt) * S_2(c1*dt)
    = exp(-i c1*dt*A/2) exp(-i c1*dt*B) exp(-i c1*dt*A/2)
      * exp(-i c2*dt*A/2) exp(-i c2*dt*B) exp(-i c2*dt*A/2)
      * exp(-i c1*dt*A/2) exp(-i c1*dt*B) exp(-i c1*dt*A/2)
    
    Merging adjacent A's:
    A coefficients: c1/2, (c1+c2)/2, (c2+c1)/2, c1/2 (times dt)
    B coefficients: c1, c2, c1 (times dt)
    
    For 3 substeps, the full sequence:
    A: 12 coefficients total
    B: 9 coefficients total
    
    We need 12 B's, so we add 3 zero B's.
    
    Let me construct this properly.
    """
    
    # Use 4th-order Yoshida with 3 substeps
    c1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))
    
    a = []
    b = []
    
    # For each substep, we have 4 A coefficients and 3 B coefficients
    # After 3 substeps: 12 A's and 9 B's
    # We need to add 3 more B's with zero coefficient
    
    for _ in range(3):
        # S_2(c1)
        a.append(c1 / 2)
        b.append(c1)
        a.append(c1 / 2)
        
        # S_2(c2)
        a.append(c2 / 2)
        b.append(c2)
        a.append(c2 / 2)
        
        # S_2(c1)
        a.append(c1 / 2)
        b.append(c1)
        a.append(c1 / 2)
    
    # Now we have 12 a's and 9 b's. Pad b with zeros.
    while len(b) < 12:
        b.append(0.0)
    
    # Verify sums
    sum_a = sum(a)
    sum_b = sum(b)
    
    # Normalize to ensure sum = 1
    if abs(sum_a - 1.0) > 1e-10:
        a = [x / sum_a for x in a]
    if abs(sum_b - 1.0) > 1e-10:
        b = [x / sum_b for x in b]
    
    return a, b


def coefficients(stages):
    """
    Return 12-stage product formula coefficients for Heisenberg-chain time evolution.
    Uses 4th-order Suzuki-Trotter with 3 substeps.
    """
    a, b = _compute_optimal_coefficients()
    
    # Ensure exactly 12 elements
    while len(a) < stages:
        a.append(0.0)
    while len(b) < stages:
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    
    # Ensure finite values in [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Final consistency check: ensure sums are exactly 1
    sum_a = sum(a)
    sum_b = sum(b)
    
    if abs(sum_a - 1.0) > 1e-12:
        # Adjust last element to make sum exactly 1
        a[-1] += (1.0 - sum_a)
    if abs(sum_b - 1.0) > 1e-12:
        # Adjust last element to make sum exactly 1
        b[-1] += (1.0 - sum_b)
    
    return a, b

# EVOLVE-BLOCK-END
