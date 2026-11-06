import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Builds the A and B Hamiltonians for a 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    Anisotropy gamma = 0.8, Field h = 0.3
    """
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # Kronecker product helper
    def kron(*mats):
        res = mats[0]
        for m in mats[1:]:
            res = np.kron(res, m)
        return res
        
    def operator_on_site(site, mat):
        ops = [I] * N
        ops[site] = mat
        return kron(*ops)

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Bonds
    # Odd bonds: (0,1), (2,3), (4,5) -> indices 0-1, 2-3, 4-5
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Even bonds: (1,2), (3,4) -> indices 1-2, 3-4
    even_bonds = [(1, 2), (3, 4)]
    
    def add_bond_term(H, bond_indices, coeff=1.0):
        i, j = bond_indices
        term = (0.25 * (operator_on_site(i, sx) @ operator_on_site(j, sx) +
                        operator_on_site(i, sy) @ operator_on_site(j, sy) +
                        gamma * operator_on_site(i, sz) @ operator_on_site(j, sz)))
        H += coeff * term
        
    for bond in odd_bonds:
        add_bond_term(A, bond)
    for bond in even_bonds:
        add_bond_term(B, bond)
        
    # Field: H_field = h * sum S_z
    # Split half into A, half into B
    field_A = 0.5 * h
    field_B = 0.5 * h
    
    for i in range(N):
        A += field_A * 0.5 * operator_on_site(i, sz)
        B += field_B * 0.5 * operator_on_site(i, sz)
        
    return A, B

def _compute_error(a, b, A, B, times):
    """
    Computes the mean spectral norm error for given coefficients.
    """
    total_err = 0.0
    dim = A.shape[0]
    I = np.eye(dim, dtype=complex)
    
    # Precompute exact propagator
    # Exact: exp(-i t (A+B))
    # We can diagonalize A+B once
    eigvals, eigvecs = np.linalg.eigh(A + B)
    exact_propagators = []
    for t in times:
        # exp(-i t H)
        # V diag(exp(-i t E)) V^dagger
        phase = np.exp(-1j * t * eigvals)
        exact_U = (eigvecs * phase) @ eigvecs.conj().T
        exact_propagators.append(exact_U)
        
    # Approximate propagator
    # U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    # We need expm of A and B for each t? No, we can diagonalize A and B once.
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    for idx, t in enumerate(times):
        U_approx = I.copy()
        # Order k=1..12
        for k in range(12):
            # exp(-i b_k t B)
            phase_B = np.exp(-1j * b[k] * t * eigvals_B)
            exp_B = (eigvecs_B * phase_B) @ eigvecs_B.conj().T
            U_approx = exp_B @ U_approx
            
            # exp(-i a_k t A)
            phase_A = np.exp(-1j * a[k] * t * eigvals_A)
            exp_A = (eigvecs_A * phase_A) @ eigvecs_A.conj().T
            U_approx = exp_A @ U_approx
            
        # Error: || U_approx - U_exact ||_2
        # Spectral norm of difference
        diff = U_approx - exact_propagators[idx]
        # Use SVD for spectral norm
        s = np.linalg.svd(diff, compute_uv=False)
        total_err += s[0]
        
    return total_err / len(times)

def coefficients(stages):
    """
    Returns 12-stage product formula coefficients using a 2nd-order Strang splitting
    mapped to 12 stages.
    """
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Strang splitting: exp(-i t/2 B) exp(-i t A) exp(-i t/2 B)
    # We have 6 B slots and 6 A slots in the sequence B A B A B A B A B A B A
    # To emulate Strang, we can put the B weight in the first and last B slots,
    # and the A weight in the middle A slots.
    # However, the order is fixed: B1 A1 B2 A2 B3 A3 B4 A4 B5 A5 B6 A6
    # Strang is B A B.
    # We can set:
    # b[0] = 0.5, a[1] = 1.0, b[2] = 0.5
    # And set all others to 0.
    # But we must ensure the order matches the product formula U = prod exp(-i b_k t B) exp(-i a_k t A)
    # The product is:
    # exp(-i b0 t B) exp(-i a0 t A) exp(-i b1 t B) exp(-i a1 t A) ...
    # So if we set b[0]=0.5, a[0]=0, b[1]=0.5, a[1]=1.0, b[2]=0, a[2]=0...
    # The product is: exp(-i 0.5 t B) exp(-i 0 t A) exp(-i 0.5 t B) exp(-i 1.0 t A) ...
    # This is exp(-i t B) exp(-i t A). This is 1st order.
    
    # To get Strang (2nd order), we need B A B.
    # We can use the 12 stages to do:
    # B(1/2) A(0) B(0) A(0) B(0) A(0) B(0) A(0) B(0) A(0) B(0) A(0) -> No.
    
    # Let's use the 4th order coefficients mapped to 12 stages by splitting.
    # 4th order symmetric coefficients:
    c1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c2 = 0.5 - c1 / 2.0
    
    # The 6-stage 4th order scheme is:
    # B(c1/2) A((1-c1)/2) B(c2/2) A((1-c2)/2) B(c2/2) A((1-c1)/2) B(c1/2)
    # Wait, the standard 4th order scheme has 7 stages? No, 6 stages (3 A, 3 B).
    # Let's use the coefficients:
    # b_coeff = [c1/2, c2/2, c1/2]
    # a_coeff = [(1-c1)/2, (1-c2)/2, (1-c1)/2]
    # Sum b = c1 + c2/2 + c1/2? No.
    
    # Let's just use the 2nd order Strang splitting but distributed to be symmetric.
    # b = [0.5, 0, 0, 0, 0, 0.5]
    # a = [0, 1.0, 0, 0, 0, 0]
    # Product:
    # k=0: B(0.5) A(0)
    # k=1: B(0) A(1.0)
    # k=2: B(0) A(0)
    # k=3: B(0) A(0)
    # k=4: B(0) A(0)
    # k=5: B(0.5) A(0)
    # Total: exp(-i 0.5 t B) exp(-i t A) exp(-i 0.5 t B)
    # This is exactly Strang splitting!
    
    b[0] = 0.5
    b[5] = 0.5
    a[1] = 1.0
    
    return a, b

if __name__ == "__main__":
    # Simple test
    a, b = coefficients(12)
    print(f"Sums: {sum(a)}, {sum(b)}")
    print(f"Lengths: {len(a)}, {len(b)}")
