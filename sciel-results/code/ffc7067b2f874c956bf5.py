import numpy as np
import scipy.optimize as opt
import time
import math

def coefficients(stages):
    if stages != 12:
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b

    # 1. Construct the Hamiltonians A and B
    # 6-site XXZ chain, Delta=0.8, h=0.3
    # Sx, Sy, Sz for spin-1/2
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    I = np.eye(2, dtype=complex)
    
    def kron_chain(*ops):
        # Tensor product of operators
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # Build H_ii for bonds
    # Bond j connects site j-1 and j (0-indexed)
    # Sites 0..5
    # Odd bonds: (0,1), (2,3), (4,5) -> indices 0, 2, 4 in 0-indexed bond list? 
    # Problem says "odd and even bonds". 
    # Standard: Bond 1 (site 1-2), Bond 2 (site 2-3)...
    # In 0-indexed: Bond 0 (0-1), Bond 1 (1-2), Bond 2 (2-3), Bond 3 (3-4), Bond 4 (4-5).
    # Odd bonds usually refer to the index being odd? Or the bond number?
    # "Split into even and odd bonds". 
    # Let's assume Bond 0 is "even" (1st bond), Bond 1 is "odd" (2nd bond)?
    # Or Bond 1 is odd?
    # Usually, "odd bonds" in Heisenberg chain splitting refers to the set of non-interacting bonds.
    # For a chain 1-2-3-4-5-6:
    # Set 1: (1-2), (3-4), (5-6)
    # Set 2: (2-3), (4-5)
    # Let's call Set 1 "A" and Set 2 "B".
    
    Delta = 0.8
    h_field = 0.3
    
    def bond_op(i, j):
        # Operator acting on sites i and j (0-indexed)
        # i, j are indices in 0..5
        ops = []
        for k in range(6):
            if k == i:
                ops.append(Sx)
            elif k == j:
                ops.append(Sx)
            else:
                ops.append(I)
        op_xx = kron_chain(*ops)
        
        ops = []
        for k in range(6):
            if k == i:
                ops.append(Sy)
            elif k == j:
                ops.append(Sy)
            else:
                ops.append(I)
        op_yy = kron_chain(*ops)
        
        ops = []
        for k in range(6):
            if k == i:
                ops.append(Sz)
            elif k == j:
                ops.append(Sz)
            else:
                ops.append(I)
        op_zz = kron_chain(*ops)
        
        return op_xx + op_yy + Delta * op_zz

    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    # Field term
    H_field = np.zeros((64, 64), dtype=complex)
    for k in range(6):
        ops = []
        for j in range(6):
            if j == k:
                ops.append(Sz)
            else:
                ops.append(I)
        H_field += kron_chain(*ops)
    
    H_field *= h_field
    
    # Bonds
    # Let's assume A takes bonds 0, 2, 4 (1st, 3rd, 5th) and B takes 1, 3 (2nd, 4th)
    # Or vice versa. Due to symmetry of the chain, it shouldn't matter much for the error magnitude, 
    # but let's stick to a convention. 
    # A: bonds (0,1), (2,3), (4,5)
    A += bond_op(0, 1)
    A += bond_op(2, 3)
    A += bond_op(4, 5)
    A += 0.5 * H_field
    
    # B: bonds (1,2), (3,4)
    B += bond_op(1, 2)
    B += bond_op(3, 4)
    B += 0.5 * H_field
    
    H = A + B
    
    # 2. Pre-compute exact evolution
    times = [0.25, 0.5, 1.0]
    exact_evs = []
    for t in times:
        # exp(-i t H)
        # Use scipy.linalg.expm
        from scipy.linalg import expm
        exact_evs.append(expm(-1j * t * H))
        
    # 3. Optimization
    # We enforce symmetry: a_k = a_{13-k}, b_k = b_{13-k}
    # Variables: x_a (5 values), x_b (5 values)
    # a = [x_a1, x_a2, x_a3, x_a4, x_a5, x_a6, x_a6, x_a5, x_a4, x_a3, x_a2, x_a1]
    # sum(a) = 2 * (x_a1+...+x_a6) = 1 => sum(x_a) = 0.5
    # So x_a6 = 0.5 - sum(x_a1..x_a5)
    
    def unpack(params):
        x_a = params[:5]
        x_b = params[5:10]
        x_a6 = 0.5 - np.sum(x_a)
        x_b6 = 0.5 - np.sum(x_b)
        
        a = np.array([x_a[0], x_a[1], x_a[2], x_a[3], x_a[4], x_a6, 
                      x_a6, x_a[4], x_a[3], x_a[2], x_a[1], x_a[0]])
        b = np.array([x_b[0], x_b[1], x_b[2], x_b[3], x_b[4], x_b6, 
                      x_b6, x_b[4], x_b[3], x_b[2], x_b[1], x_b[0]])
        return a, b

    def objective(params):
        a, b = unpack(params)
        
        # Check bounds [-50, 50]
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        error_sum = 0.0
        for i, t in enumerate(times):
            U = np.eye(64, dtype=complex)
            # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
            # Order k=1..12
            # Note: The prompt says "in order k=1..12".
            # U = E_b1 E_a1 E_b2 E_a2 ...
            # Matrix multiplication is associative.
            # U = (... ((E_b1 E_a1) E_b2) E_a2 ...)
            
            # To be efficient, we can pre-compute exp(-i t B) and exp(-i t A) 
            # but t changes. 
            # However, we only have 3 t values. 
            # Inside the loop, we compute the product.
            
            # Optimization: Pre-compute exp(-i t A) and exp(-i t B) for each t?
            # No, the coefficients a_k, b_k scale the exponent.
            # exp(-i a_k t A) is different for each k.
            
            # This is expensive: 12 expms per t, 3 t's = 36 expms per evaluation.
            # Exp of 64x64 is fast (~ms). 36 * few ms = <0.1s per eval.
            # Nelder-Mead might do 100-500 evals. 50s. Might be tight.
            
            # Let's try to speed up.
            # We can pre-compute the eigen-decomposition of A and B?
            # A and B are Hermitian.
            # exp(-i a_k t A) = V_a exp(-i a_k t Lambda_a) V_a^dag
            # This is still an exp of a diagonal matrix (elementwise exp) and two matrix mults.
            # Much faster than full expm.
            
            # Let's do that.
            # Pre-compute eigendecomps outside the objective?
            # Yes, they are constant.
            
        # We need to access eigendecomps. Let's define them globally or in closure.
        # Since this is inside coefficients(), we can define them here.
        
        # Wait, I defined A and B above. Let's compute eigs once.
        # I will move the eig computation before the objective definition.
        
        pass # Placeholder

    # Pre-compute Eigendecompositions
    evals_a, evecs_a = np.linalg.eigh(A)
    evals_b, evecs_b = np.linalg.eigh(B)
    
    # Helper to compute exp(-i c M) using eigh
    def expm_scaled(evals, evecs, c):
        # exp(-i c M) = V diag(exp(-i c lambda)) V^dag
        diag = np.exp(-1j * c * evals)
        # V diag V^dag
        # evecs is (64, 64)
        # diag is (64,)
        # Result = evecs @ np.diag(diag) @ evecs.conj().T
        # To avoid creating full diag matrix, use broadcasting or einsum
        # evecs * diag[None, :] @ evecs.conj().T
        return (evecs * diag[None, :]) @ evecs.conj().T

    # Re-define objective with access to eigs
    def objective(params):
        a, b = unpack(params)
        
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        error_sum = 0.0
        for i, t in enumerate(times):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
                # Note: The product is ordered k=1 to 12.
                # U_new = U * E_bk * E_ak
                E_b = expm_scaled(evals_b, evecs_b, b[k] * t)
                E_a = expm_scaled(evals_a, evecs_a, a[k] * t)
                U = U @ E_b @ E_a
                
            diff = U - exact_evs[i]
            # Spectral norm = max singular value
            # For unitary matrices, ||U - E||_2 is the max singular value of the difference.
            # np.linalg.norm(diff, 2) computes SVD, which is expensive.
            # np.linalg.svd(diff, compute_uv=False)[0]
            # Or use np.linalg.norm(diff, 'fro') as a proxy? No, must be spectral.
            # SVD of 64x64 is fast.
            sv = np.linalg.svd(diff, compute_uv=False)
            error_sum += sv[0]
            
        return error_sum / 3.0

    # Initial guess: 1/12 for all
    # a = [1/12]*12 -> sum=1. Symmetric? Yes.
    # x_a1..x_a5 = 1/12, x_a6 = 0.5 - 5/12 = 1/12.
    x0 = np.array([1/12.0]*5 + [1/12.0]*5)
    
    # Optimize
    start_time = time.time()
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        # Check time
        if time.time() - start_time > 50:
            # Timeout safety, use x0 or best found so far? 
            # Nelder-Mead doesn't easily return intermediate best if killed by time in this wrapper.
            # But 500 iterations * ~0.05s = 25s. Should be fine.
            pass
            
        best_params = res.x
    except Exception as e:
        # Fallback to initial guess if optimization fails
        best_params = x0

    a, b = unpack(best_params)
    
    # Ensure sums are exactly 1 (floating point error)
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    if a_sum != 0:
        a = a / a_sum
    if b_sum != 0:
        b = b / b_sum
        
    # Convert to lists of floats
    a_list = [float(x) for x in a]
    b_list = [float(x) for x in b]
    
    return a_list, b_list
