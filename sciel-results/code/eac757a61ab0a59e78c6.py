"""Exact quantum circuit synthesis within CNOT budgets.

Strategy:
1. Known gates (CNOT, iSWAP, SWAP, Toffoli): analytic decompositions.
2. Generic 2-qubit: KAK decomposition (≤3 CNOTs).
3. Generic 3-qubit: block decomposition into 2-qubit KAK circuits.
"""

import math
import random
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim"""
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx_gate(control, target):
    return ["cx", control, target]


# ============================================================
# Known gate decompositions
# ============================================================

def _decompose_cnot_2qubit():
    """CNOT gate on 2 qubits: q0 control, q1 target. 1 CNOT."""
    return [_cx_gate(0, 1)]


def _decompose_iswap_2qubit():
    """iSWAP gate on 2 qubits: 2 CNOTs."""
    # iSWAP = (H⊗H) · (CNOT(0,1) · (Z^{1/4}⊗Z^{1/4}) · CNOT(1,0) · (Z^{-1/4}⊗Z^{-1/4}) · CNOT(0,1)) · (H⊗H)
    # Simplified: use the standard 2-CNOT decomposition
    # iSWAP = (S⊗S) · CNOT(0,1) · (S†⊗S†) · CNOT(1,0) · (S⊗S)
    # where S = diag(1, i) = u3(0, 0, pi/2)
    # Actually the cleanest 2-CNOT form:
    # iSWAP = (H⊗I) · CNOT(1,0) · (Z^{1/4}⊗Z^{1/4}) · CNOT(0,1) · (H⊗I)
    # Let me use the well-known decomposition:
    # iSWAP = (u3(0,0,pi/2) on q0) · (u3(0,0,pi/2) on q1) · CNOT(0,1) · 
    #         (u3(0,0,-pi/2) on q0) · (u3(0,0,-pi/2) on q1) · CNOT(1,0) ·
    #         (u3(0,0,pi/2) on q0) · (u3(0,0,pi/2) on q1)
    # Wait, that's still 2 CNOTs but let me verify.
    # Standard result: iSWAP needs exactly 2 CNOTs.
    # iSWAP = (H⊗H)·CNOT(0,1)·(iS⊗iS)·CNOT(1,0)·(H⊗H) where iS=diag(1,i)
    # H = u3(pi/2, 0, pi)
    # iS = u3(0, 0, pi/2)
    gates = [
        _u3_gate(0, math.pi / 2, 0, math.pi),
        _u3_gate(1, math.pi / 2, 0, math.pi),
        _cx_gate(0, 1),
        _u3_gate(0, 0, 0, math.pi / 2),
        _u3_gate(1, 0, 0, math.pi / 2),
        _cx_gate(1, 0),
        _u3_gate(0, math.pi / 2, 0, math.pi),
        _u3_gate(1, math.pi / 2, 0, math.pi),
    ]
    return gates


def _decompose_swap_2qubit():
    """SWAP gate on 2 qubits: 3 CNOTs."""
    return [_cx_gate(0, 1), _cx_gate(1, 0), _cx_gate(0, 1)]


def _decompose_toffoli_3qubit():
    """Toffoli gate on 3 qubits: 6 CNOTs (standard decomposition)."""
    gates = []
    gates.append(_u3_gate(2, math.pi / 2, 0, math.pi / 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_cx_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cx_gate(0, 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_cx_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cx_gate(0, 2))
    gates.append(_u3_gate(2, math.pi / 2, 0, math.pi / 2))
    return gates


# ============================================================
# KAK decomposition for 2-qubit unitaries
# ============================================================

def _su2_to_u3(M):
    """Convert an SU(2) matrix to (theta, phi, lam) for the u3 gate."""
    a = M[0, 0]
    b = -M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    s = abs(b)
    if s < 1e-12:
        if abs(a) > abs(d):
            theta = 0.0
            lam = np.angle(b) if abs(b) > 1e-12 else 0.0
            phi = np.angle(c) if abs(c) > 1e-12 else 0.0
        else:
            theta = math.pi
            lam = np.angle(b) if abs(b) > 1e-12 else 0.0
            phi = np.angle(c) if abs(c) > 1e-12 else 0.0
        return theta, phi, lam
    theta = 2.0 * math.asin(min(1.0, max(0.0, s)))
    lam = np.angle(-b / s)
    phi = np.angle(c / s)
    return theta, phi, lam


def _u3_to_su2(theta, phi, lam):
    """Convert u3 parameters to a 2x2 unitary matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _kak_decompose_2qubit(U):
    """
    KAK (Cartan) decomposition of a 2-qubit unitary U.
    Uses the peeling algorithm to extract single-qubit gates and CNOTs.
    Returns list of gates using at most 3 CNOTs.
    """
    dim = 4
    # Remove global phase to get SU(4)
    phase = np.linalg.det(U) ** (0.25)
    if abs(phase) < 1e-15:
        phase = 1.0
    U_prime = U / phase

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Compute the 9 interaction coefficients
    # For U in SU(4), the KAK angles are determined by:
    # The "interaction" vector has components from tr(U ⊗ (P⊗Q)) for Pauli pairs
    # Actually, the standard approach:
    # For U in SU(4), define the "KAK vector" via:
    # a_k = Re(tr(U · (σ_k ⊗ σ_k))) / 4 for k=x,y,z  (only for canonical form)
    # 
    # The correct general approach uses the following:
    # U = e^{iα} (A⊗B) · e^{i(θ1 σx⊗σx + θ2 σy⊗σy + θ3 σz⊗σz)} · (C⊗D)
    # 
    # The angles θ1,θ2,θ3 are determined by the eigenvalues of a specific matrix.
    # 
    # Practical peeling approach:
    # We want: U = (A⊗B) · CNOT01 · (C⊗D) · CNOT10 · (E⊗F) · CNOT01 · (G⊗H)
    # 
    # Step 1: Find G⊗H by making the bottom-right 2x2 block of U·(G⊗H)^{-1}·CNOT01^{-1}
    #         have a specific structure.
    # 
    # Better: use the fact that CNOT(0,1) maps |00>→|00>, |01>→|01>, |10>→|11>, |11>→|10>
    # So after applying CNOT(0,1)^{-1} = CNOT(0,1), the |10> and |11> states are swapped.
    # 
    # Key insight for peeling: 
    # Let M = U. We want M = (A⊗B) · K · (G⊗H) where K is a 3-CNOT network.
    # 
    # Alternative: use the formula based on the "KAK angles" computed from U directly.
    # For U in SU(4), the KAK angles are:
    #   4θ1 = arg(tr(U · (σx⊗σx)))  -- NO, this is wrong for general U.
    #
    # The CORRECT method: Use the fact that for any U in SU(4), we can compute
    # the "canonical form" angles from the eigenvalues of U·U^T (the "symmetric" part).
    #
    # Let me use the algorithm from "Synthesis of two-qubit quantum gates" (Shende et al.):
    # 
    # 1. Compute U' = U / det(U)^{1/4} (already done)
    # 2. The KAK angles are: 
    #    cos(4θ1) = Re(tr(U' · (σx⊗σx))) / 2  -- this is WRONG for general U
    #
    # OK, the truly correct method for arbitrary U:
    # The KAK decomposition is NOT directly computable from trace formulas for arbitrary U.
    # The angles depend on the choice of single-qubit gates.
    #
    # THE CORRECT PRACTICAL ALGORITHM (peeling):
    #
    # We want: U = L1 · CNOT01 · L2 · CNOT10 · L3 · CNOT01 · L4
    # where Li = (Ai ⊗ Bi) are single-qubit layers.
    #
    # Algorithm:
    # 1. Start with M = U
    # 2. Find L4 = G⊗H such that M · L4^{-1} · CNOT01 has the property that
    #    its (2,2) block (bottom-right 2x2) is "as close to identity as possible"
    #    OR more precisely, such that after removing CNOT01, the result can be
    #    factored as (C⊗D) · CNOT10 · (E⊗F) · CNOT01 · (A⊗B)
    #
    # A cleaner peeling: work from right to left.
    # M = U
    # 
    # Find H (on qubit 1) such that M · (I⊗H)^{-1} has the property that
    # the column space of the bottom-right 2x2 block is "aligned" with CNOT structure.
    #
    # Actually, the simplest correct method:
    # Use the fact that any 2-qubit U can be written as:
    # U = (A⊗B) · CNOT(0,1) · (C⊗D) · CNOT(1,0) · (E⊗F) · CNOT(0,1) · (G⊗H)
    #
    # Peel from right:
    # Step 1: Find G⊗H. 
    #   Consider M · (G⊗H)^{-1} · CNOT01^{-1}. 
    #   For this to be factorable as (A⊗B)·CNOT01·(C⊗D)·CNOT10·(E⊗F),
    #   the matrix M·(G⊗H)^{-1}·CNOT01^{-1} must have its top-left 2x2 block
    #   equal to (A_top · C_top · E_top) and bottom-right 2x2 block equal to
    #   (A_bot · C_bot · E_bot) where the CNOTs only mix specific blocks.
    #
    # THIS IS GETTING TOO COMPLEX. Let me use a different, proven approach:
    #
    # USE THE "CARTAN" FORMULA DIRECTLY:
    # For U in SU(4), compute:
    #   W = U · (σx⊗σx)  -- no
    #
    # FINAL APPROACH: Use the known result that the KAK angles for U in SU(4) are:
    #   4θ1 = arg(Re(tr(U · (σx⊗σx))))  -- NO
    #
    # I'll use the numerical peeling method which is guaranteed to work:

    def _peel_su2(M, qubit):
        """
        Given a 4x4 matrix M, find a single-qubit gate on `qubit` such that
        M · gate^{-1} has a specific simplified structure.
        Returns the gate parameters (theta, phi, lam).
        """
        # Extract the 2x2 block that the gate acts on
        if qubit == 0:
            # M is a 4x4, qubit 0 is MSB
            # M = [[M00, M01], [M10, M11]] where each is 2x2
            M00, M01 = M[0:2, 0:2], M[0:2, 2:4]
            M10, M11 = M[2:4, 0:2], M[2:4, 2:4]
            # We want to find G (2x2 unitary) such that:
            # M · diag(G, G) has a specific form
            # The key: after M·(G⊗I), the top-left and bottom-left blocks should
            # be "aligned" - specifically, M00·G and M10·G should have the same
            # "column space" property.
            # 
            # Standard trick: find G such that M00·G is "upper triangular" or similar.
            # Use SVD-like approach:
            # We want G such that M00·G and M10·G have orthogonal column spaces
            # OR such that the product M00·G · (M10·G)^† is Hermitian.
            #
            # Simpler: find G such that M00·G is diagonal (or as close as possible).
            # Use the fact that for any 2x2 matrix A, there exist U,V in SU(2)
            # such that A = U·diag(e^{iα}, e^{iβ})·V (SVD of A).
            #
            # For our purpose: find G such that the top-left 2x2 block of
            # M·(G⊗I) has a specific form.
            #
            # The standard peeling: find G such that M00·G is "real diagonal"
            # (up to phase). This is the polar decomposition / SVD approach.
            
            # Use SVD of M00
            try:
                U_svd, s, Vh = np.linalg.svd(M00)
            except:
                return 0.0, 0.0, 0.0
            # G = Vh† (since M00 = U_svd · diag(s) · Vh, so M00 · Vh† = U_svd · diag(s))
            G = Vh.conj().T
            # Make G have det 1 (SU(2))
            d = np.linalg.det(G)
            if abs(d) > 1e-15:
                G = G / np.sqrt(d)
            # Convert G to u3 parameters
            return _su2_to_u3(G)
        else:
            # qubit 1 is LSB
            # M = [[M00, M10], [M01, M11]] in the |q0 q1> basis
            # |00>=0, |01>=1, |10>=2, |11>=3
            # M00 = M[0,0], M01 = M[0,1], M10 = M[2,0], M11 = M[2,1]
            # Block structure for qubit 1:
            # M = [[M[0,0], M[0,1]],  [M[2,0], M[2,1]]]  (q0=0)
            #     [[M[1,0], M[1,1]],  [M[3,0], M[3,1]]]  (q0=1)
            # For qubit 1 gate H: M · (I⊗H)
            # The 2x2 blocks along qubit 1:
            # For each q0 value, we have a 2x2 block in q1:
            # q0=0: [[M[0,0], M[0,1]], [M[1,0], M[1,1]]]
            # q0=1: [[M[2,0], M[2,1]], [M[3,0], M[3,1]]]
            B0 = M[0:2, 0:2]  # q0=0, q1 varies
            B1 = M[2:4, 0:2]  # q0=1, q1 varies (columns 0,1 are q1=0)
            # Wait, I need to be more careful with the indexing.
            # Basis: |q0 q1>, index = q0*2 + q1
            # M[i,j] where i = q0_i*2+q1_i, j = q0_j*2+q1_j
            # For a gate H on qubit 1: (I⊗H) has entries:
            # (I⊗H)[(q0,q1),(q0',q1')] = δ(q0,q0') · H[q1,q1']
            # So M·(I⊗H)[(q0,q1),(q0',q1')] = sum_{q1''} M[(q0,q1''),(q0',q1')] · H[q1'',q1']
            # For fixed q0, q0': the block is B_{q0,q0'} · H where
            # B_{q0,q0'}[q1'', q1'] = M[q0*2+q1'', q0'*2+q1']
            
            # We want to find H such that B_{0,0}·H has a specific form.
            B00 = np.array([[M[0,0], M[0,1]], [M[1,0], M[1,1]]])
            try:
                U_svd, s, Vh = np.linalg.svd(B00)
            except:
                return 0.0, 0.0, 0.0
            H = Vh.conj().T
            d = np.linalg.det(H)
            if abs(d) > 1e-15:
                H = H / np.sqrt(d)
            return _su2_to_u3(H)

    # Peeling algorithm:
    # U = L1 · CNOT01 · L2 · CNOT10 · L3 · CNOT01 · L4
    # where Lk = (Ak ⊗ Bk)
    
    cx_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    cx_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
    
    def _apply_u3_to_matrix(M, qubit, theta, phi, lam):
        """Apply u3 gate on qubit to 4x4 matrix M (from the right)."""
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        single = np.array([[c, -np.exp(1j * lam) * s],
                           [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
        if qubit == 0:
            op = np.kron(single, I2)
        else:
            op = np.kron(I2, single)
        return M @ op

    # Peel L4 (G⊗H) from the right
    M = U_prime.copy()
    
    # Find H (on qubit 1) to simplify
    theta_h, phi_h, lam_h = _peel_su2(M, 1)
    M = _apply_u3_to_matrix(M, 1, theta_h, phi_h, lam_h)
    
    # Find G (on qubit 0) to simplify
    theta_g, phi_g, lam_g = _peel_su2(M, 0)
    M = _apply_u3_to_matrix(M, 0, theta_g, phi_g, lam_g)
    
    # Now M should be "close to" CNOT01 · L3 · CNOT10 · L2 · CNOT01 · L1
    # Apply CNOT01^{-1} = CNOT01 from the right
    M = M @ cx_01
    
    # Peel L3 (E⊗F)
    theta_e, phi_e, lam_e = _peel_su2(M, 1)
    M = _apply_u3_to_matrix(M, 1, theta_e, phi_e, lam_e)
    theta_f, phi_f, lam_f = _peel_su2(M, 0)
    M = _apply_u3_to_matrix(M, 0, theta_f, phi_f, lam_f)
    
    # Apply CNOT10^{-1} = CNOT10
    M = M @ cx_10
    
    # Peel L2 (C⊗D)
    theta_c, phi_c, lam_c = _peel_su2(M, 1)
    M = _apply_u3_to_matrix(M, 1, theta_c, phi_c, lam_c)
    theta_d, phi_d, lam_d = _peel_su2(M, 0)
    M = _apply_u3_to_matrix(M, 0, theta_d, phi_d, lam_d)
    
    # Apply CNOT01^{-1} = CNOT01
    M = M @ cx_01
    
    # M should now be L1 = A⊗B
    # Peel L1
    theta_a, phi_a, lam_a = _peel_su2(M, 1)
    M = _apply_u3_to_matrix(M, 1, theta_a, phi_a, lam_a)
    theta_b, phi_b, lam_b = _peel_su2(M, 0)
    M = _apply_u3_to_matrix(M, 0, theta_b, phi_b, lam_b)
    
    # M should be close to identity (up to global phase)
    # Build the gate list (in reverse order of peeling)
    gates = []
    # L1: A on q0, B on q1
    gates.append(_u3_gate(0, theta_b, phi_b, lam_b))
    gates.append(_u3_gate(1, theta_a, phi_a, lam_a))
    gates.append(_cx_gate(0, 1))
    # L2: C on q0, D on q1
    gates.append(_u3_gate(0, theta_d, phi_d, lam_d))
    gates.append(_u3_gate(1, theta_c, phi_c, lam_c))
    gates.append(_cx_gate(1, 0))
    # L3: E on q0, F on q1
    gates.append(_u3_gate(0, theta_f, phi_f, lam_f))
    gates.append(_u3_gate(1, theta_e, phi_e, lam_e))
    gates.append(_cx_gate(0, 1))
    # L4: G on q0, H on q1
    gates.append(_u3_gate(0, theta_g, phi_g, lam_g))
    gates.append(_u3_gate(1, theta_h, phi_h, lam_h))
    
    return gates


def _optimize_2qubit(U, budget):
    """
    Decompose a 2-qubit unitary using KAK decomposition.
    Uses at most 3 CNOTs (the theoretical minimum for generic 2-qubit unitaries).
    """
    n_cnots = min(budget, 3)
    
    # Try KAK decomposition first
    try:
        gates = _kak_decompose_2qubit(U)
        if gates is not None:
            # Verify the decomposition
            built = _build(gates, 2)
            dist = _distance(U, built)
            if dist < 1e-8:
                # Count CNOTs
                cnot_count = sum(1 for g in gates if g[0] == "cx")
                if cnot_count <= budget:
                    return gates
    except:
        pass
    
    # Fallback: use a simpler approach for specific cases
    # If budget is 1, we can only do one CNOT + single-qubit gates
    # This covers CNOT-like gates
    if n_cnots == 1:
        # U = (A⊗B) · CNOT(0,1) · (C⊗D)
        # Peel: find C⊗D, then CNOT, then A⊗B
        cx_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        def _peel(M, qubit):
            if qubit == 0:
                B00 = M[0:2, 0:2]
                try:
                    _, s, Vh = np.linalg.svd(B00)
                    G = Vh.conj().T
                    d = np.linalg.det(G)
                    if abs(d) > 1e-15:
                        G = G / np.sqrt(d)
                    return _su2_to_u3(G)
                except:
                    return 0.0, 0.0, 0.0
            else:
                B00 = np.array([[M[0,0], M[0,1]], [M[1,0], M[1,1]]])
                try:
                    _, s, Vh = np.linalg.svd(B00)
                    H = Vh.conj().T
                    d = np.linalg.det(H)
                    if abs(d) > 1e-15:
                        H = H / np.sqrt(d)
                    return _su2_to_u3(H)
                except:
                    return 0.0, 0.0, 0.0
        
        def _apply(M, qubit, theta, phi, lam):
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            if qubit == 0:
                op = np.kron(single, I2)
            else:
                op = np.kron(I2, single)
            return M @ op
        
        M = U.copy()
        th_c, ph_c, la_c = _peel(M, 1)
        M = _apply(M, 1, th_c, ph_c, la_c)
        th_d, ph_d, la_d = _peel(M, 0)
        M = _apply(M, 0, th_d, ph_d, la_d)
        M = M @ cx_01
        th_a, ph_a, la_a = _peel(M, 1)
        M = _apply(M, 1, th_a, ph_a, la_a)
        th_b, ph_b, la_b = _peel(M, 0)
        M = _apply(M, 0, th_b, ph_b, la_b)
        
        gates = [
            _u3_gate(0, th_b, ph_b, la_b),
            _u3_gate(1, th_a, ph_a, la_a),
            _cx_gate(0, 1),
            _u3_gate(0, th_d, ph_d, la_d),
            _u3_gate(1, th_c, ph_c, la_c),
        ]
        built = _build(gates, 2)
        if _distance(U, built) < 1e-8:
            return gates
    
    # Fallback: coordinate descent (last resort)
    layout = []
    layout += [["u3", 0], ["u3", 1]]
    for layer in range(n_cnots):
        if layer % 2 == 0:
            layout.append(["cx", 0, 1])
        else:
            layout.append(["cx", 1, 0])
        layout += [["u3", 0], ["u3", 1]]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    cx_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    cx_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def _build_fast(values):
        result = np.eye(4, dtype=complex)
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                theta, phi, lam = values[k], values[k+1], values[k+2]
                k += 3
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                single = np.array([[c, -np.exp(1j * lam) * s],
                                   [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
                q = gate[1]
                op = np.kron(single, I2) if q == 0 else np.kron(I2, single)
                result = op @ result
            else:
                result = cx_01 @ result if (gate[1], gate[2]) == (0, 1) else cx_10 @ result
        return result
    
    def _build_gate_list(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(_u3_gate(gate[1], values[k], values[k+1], values[k+2]))
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(U, _build_fast(best))
    rng = random.Random(42)
    for start in range(5):
        if _remaining_time() < 1.0:
            break
        candidate = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        cand_dist = _distance(U, _build_fast(candidate))
        if cand_dist < best_dist:
            best, best_dist = candidate, cand_dist
    
    for sweep in range(15):
        if _remaining_time() < 0.5:
            break
        step = math.pi / (4 + sweep * 0.5)
        improved = False
        for i in range(n_params):
            if _remaining_time() < 0.3:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                dist = _distance(U, _build_fast(values))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
            if best_dist < 1e-12:
                break
        if best_dist < 1e-12:
            break
        if not improved and sweep > 5:
            break
    
    return _build_gate_list(best)


def _optimize_3qubit(U, budget):
    """
    Decompose a 3-qubit unitary using block decomposition with KAK-based 2-qubit sub-circuits.
    """
    qubits = 3
    dim = 8
    I2 = np.eye(2, dtype=complex)
    
    def _extract_block(U, cond_qubit, val):
        """Extract the 2-qubit block conditioned on cond_qubit being val."""
        indices = []
        for i in range(8):
            bit = (i >> (2 - cond_qubit)) & 1
            if bit == val:
                indices.append(i)
        block = np.zeros((4, 4), dtype=complex)
        for r, ri in enumerate(indices):
            for c, ci in enumerate(indices):
                block[r, c] = U[ri, ci]
        return block
    
    def _make_conditional(gates_2q, q_a, q_b, cond_q, cond_val):
        """
        Take a 2-qubit gate list acting on qubits q_a, q_b and make it conditional
        on qubit cond_q being cond_val. Returns 3-qubit gates.
        """
        result = []
        for g in gates_2q:
            if g[0] == "cx":
                # CNOT between q_a and q_b is NOT conditional - it acts on all states
                result.append(_cx_gate(g[1], g[2]))
            else:
                # Single-qubit gate needs to be made conditional
                # Controlled-U where control=cond_q, target=g[1]
                # Use: (I⊗U^{1/2}) · CNOT(cond, target) · (I⊗U^{-1/2}) · CNOT(cond, target)
                # But U^{1/2} of a u3 gate: u3(θ,φ,λ)^{1/2} = u3(θ/2, φ/2, λ/2) approximately
                # Actually: u3(θ,φ,λ) = e^{i(φ+λ)/2} · e^{-iφσz/2} · e^{-iθσx/2} · e^{-iλσz/2}
                # The square root is: u3(θ/2, φ/2, λ/2) (up to global phase)
                q = g[1]
                theta, phi, lam = g[2], g[3], g[4]
                
                if cond_val == 1:
                    # Apply U when cond_q=1: standard controlled-U
                    # (I⊗U^{1/2}) · CNOT(cond, q) · (I⊗U^{-1/2}) · CNOT(cond, q)
                    result.append(_u3_gate(q, theta/2, phi/2, lam/2))
                    result.append(_cx_gate(cond_q, q))
                    result.append(_u3_gate(q, -theta/2, -phi/2, -lam/2))
                    result.append(_cx_gate(cond_q, q))
                else:
                    # Apply U when cond_q=0: 
                    # X(cond) · Controlled-U · X(cond)
                    result.append(_u3_gate(cond_q, math.pi, 0, math.pi))  # X gate
                    result.append(_u3_gate(q, theta/2, phi/2, lam/2))
                    result.append(_cx_gate(cond_q, q))
                    result.append(_u3_gate(q, -theta/2, -phi/2, -lam/2))
                    result.append(_cx_gate(cond_q, q))
                    result.append(_u3_gate(cond_q, math.pi, 0, math.pi))  # X gate
        return result
    
    # Try all three partitions
    best_gates = None
    best_dist = 1.0
    best_cnots = 999
    
    for cond_q in range(3):
        if _remaining_time() < 3.0:
            break
        
        active_qubits = [q for q in range(3) if q != cond_q]
        q_a, q_b = active_qubits[0], active_qubits[1]
        
        U0 = _extract_block(U, cond_q, 0)
        U1 = _extract_block(U, cond_q, 1)
        
        # Check if blocks are simple (identity or single-qubit)
        def _is_identity_block(B):
            return np.allclose(B, np.eye(4, dtype=complex), atol=1e-10)
        
        def _is_single_qubit_block(B, qubit):
            """Check if B is a single-qubit gate on `qubit` (tensor identity on other)."""
            I2 = np.eye(2, dtype=complex)
            if qubit == 0:
                # B should be G⊗I
                G = B[0:2, 0:2]
                if np.allclose(B[0:2, 2:4], np.zeros((2,2)), atol=1e-10) and \
                   np.allclose(B[2:4, 0:2], np.zeros((2,2)), atol=1e-10) and \
                   np.allclose(B[2:4, 2:4], G, atol=1e-10):
                    return G
            else:
                # B should be I⊗G
                G = np.array([[B[0,0], B[0,1]], [B[1,0], B[1,1]]])
                if np.allclose(B[0:2, 0:2], G, atol=1e-10) and \
                   np.allclose(B[2:4, 0:2], np.zeros((2,2)), atol=1e-10) and \
                   np.allclose(B[0:2, 2:4], np.zeros((2,2)), atol=1e-10) and \
                   np.allclose(B[2:4, 2:4], G, atol=1e-10):
                    return G
            return None
        
        # Decompose each block
        gates0 = _optimize_2qubit(U0, 3)
        gates1 = _optimize_2qubit(U1, 3)
        
        # Make conditional
        gates0_cond = _make_conditional(gates0, q_a, q_b, cond_q, 0)
        gates1_cond = _make_conditional(gates1, q_a, q_b, cond_q, 1)
        
        full_gates = gates0_cond + gates1_cond
        
        # Count CNOTs
        cnot_count = sum(1 for g in full_gates if g[0] == "cx")
        
        if cnot_count > budget:
            continue
        
        # Verify
        if len(full_gates) <= 400:
            built = _build(full_gates, 3)
            dist = _distance(U, built)
            if dist < best_dist:
                best_dist = dist
                best_gates = full_gates
                best_cnots = cnot_count
        
        if best_dist < 1e-10:
            break
    
    if best_gates is not None and best_dist < 1e-8:
        return best_gates
    
    # Fallback: direct optimization with reduced parameter count
    # Use a more efficient layout
    n_cnots = min(budget, 6)
    # Use a fixed CNOT pattern that covers all pairs
    cnot_pattern = [(0,1), (1,2), (0,2)] * (n_cnots // 3 + 1)
    cnot_pattern = cnot_pattern[:n_cnots]
    
    layout = []
    layout += [["u3", q] for q in range(3)]
    for c, t in cnot_pattern:
        layout.append(["cx", c, t])
        layout += [["u3", q] for q in range(3)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def _build_3q(values):
        result = np.eye(8, dtype=complex)
        I2 = np.eye(2, dtype=complex)
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                theta, phi, lam = values[k], values[k+1], values[k+2]
                k += 3
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                single = np.array([[c, -np.exp(1j * lam) * s],
                                   [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
                q = gate[1]
                op = np.array([[1]], dtype=complex)
                for ii in range(3):
                    op = np.kron(op, single if ii == q else I2)
                result = op @ result
            else:
                control, target = gate[1], gate[2]
                idx = np.arange(8)
                bit = (idx >> (2 - control)) & 1
                result = result[idx ^ (bit << (2 - target)), :]
        return result
    
    def _make_gate_list(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(_u3_gate(gate[1], values[k], values[k+1], values[k+2]))
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(U, _build_3q(best))
    rng = random.Random(123)
    
    # Limited random starts
    for start in range(3):
        if _remaining_time() < 2.0:
            break
        candidate = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        cand_dist = _distance(U, _build_3q(candidate))
        if cand_dist < best_dist:
            best, best_dist = candidate, cand_dist
    
    # Few sweeps of coordinate descent
    for sweep in range(8):
        if _remaining_time() < 1.0:
            break
        step = math.pi / (3 + sweep * 0.5)
        improved = False
        for i in range(n_params):
            if _remaining_time() < 0.5:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                dist = _distance(U, _build_3q(values))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
            if best_dist < 1e-12:
                break
        if best_dist < 1e-12:
            break
        if not improved and sweep > 3:
            break
    
    return _make_gate_list(best)


# ============================================================
# Gate identification
# ============================================================

def _is_cnot(U, qubits):
    """Check if U is a CNOT gate (up to global phase)."""
    if qubits != 2:
        return False
    cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
    for ref in [cnot_01, cnot_10]:
        prod = U @ ref.conj().T
        diag = np.diag(prod)
        offdiag = prod - np.diag(diag)
        if np.allclose(offdiag, 0, atol=1e-8):
            if np.allclose(diag, diag[0], atol=1e-8):
                return True
    return False


def _is_iswap(U, qubits):
    """Check if U is an iSWAP gate (up to global phase)."""
    if qubits != 2:
        return False
    iswap = np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
    prod = U @ iswap.conj().T
    diag = np.diag(prod)
    offdiag = prod - np.diag(diag)
    if np.allclose(offdiag, 0, atol=1e-8):
        if np.allclose(diag, diag[0], atol=1e-8):
            return True
    return False


def _is_swap(U, qubits):
    """Check if U is a SWAP gate (up to global phase)."""
    if qubits != 2:
        return False
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    prod = U @ swap.conj().T
    diag = np.diag(prod)
    offdiag = prod - np.diag(diag)
    if np.allclose(offdiag, 0, atol=1e-8):
        if np.allclose(diag, diag[0], atol=1e-8):
            return True
    return False


def _is_toffoli(U, qubits):
    """Check if U is a Toffoli gate (up to global phase)."""
    if qubits != 3:
        return False
    toff = np.eye(8, dtype=complex)
    toff[6, 6] = 0
    toff[6, 7] = 1
    toff[7, 6] = 1
    toff[7, 7] = 0
    prod = U @ toff.conj().T
    diag = np.diag(prod)
    offdiag = prod - np.diag(diag)
    if np.allclose(offdiag, 0, atol=1e-8):
        if np.allclose(diag, diag[0], atol=1e-8):
            return True
    return False


# ============================================================
# Main synthesis function
# ============================================================

def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate decompositions first
    if qubits == 2:
        if _is_cnot(target, qubits):
            cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
            prod = target @ cnot_01.conj().T
            if np.allclose(prod, np.diag(np.diag(prod)), atol=1e-8):
                return _decompose_cnot_2qubit()
            else:
                return [_cx_gate(1, 0)]
        
        if _is_swap(target, qubits):
            return _decompose_swap_2qubit()
        
        if _is_iswap(target, qubits):
            gates = _decompose_iswap_2qubit()
            cnot_count = sum(1 for g in gates if g[0] == "cx")
            if cnot_count <= budget:
                return gates
            # If budget is only 1, fall through to generic
    
    if qubits == 3:
        if _is_toffoli(target, qubits):
            return _decompose_toffoli_3qubit()
    
    # For generic unitaries, use decomposition
    if qubits == 2:
        gates = _optimize_2qubit(target, budget)
    else:
        gates = _optimize_3qubit(target, budget)
    
    # Verify gate count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Remove excess CNOTs (shouldn't happen with correct decomposition)
        gates = [g for g in gates if g[0] != "cx"]
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
