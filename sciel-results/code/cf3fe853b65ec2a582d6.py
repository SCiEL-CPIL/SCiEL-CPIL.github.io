import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=20):
        """Return True if the decomposition exactly reproduces A@B for random inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for t in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n*m) @ Ua)
                y = (B.reshape(m*p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_exp_T = C_expected.T
                if not np.allclose(C_computed, C_exp_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def verify_exhaustive(U, V, W, seed=12345, trials=50):
        """More thorough verification with structured inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for t in range(trials):
                if t % 5 == 0:
                    A = rng.rand(n, m)
                    B = rng.rand(m, p)
                elif t % 5 == 1:
                    A = rng.randint(0, 2, (n, m)).astype(float)
                    B = rng.randint(0, 2, (m, p)).astype(float)
                elif t % 5 == 2:
                    A = np.zeros((n, m)); A[rng.randint(n), rng.randint(m)] = 1
                    B = np.zeros((m, p)); B[rng.randint(m), rng.randint(p)] = 1
                elif t % 5 == 3:
                    A = rng.randn(n, m)
                    B = rng.randn(m, p)
                else:
                    A = rng.rand(n, m)
                    B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n*m) @ Ua)
                y = (B.reshape(m*p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_exp_T = C_expected.T
                if not np.allclose(C_computed, C_exp_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def exhaustive_rank_reduce(U, V, W, max_removed=10):
        """Exhaustive search: try removing all subsets of columns up to size 2,
        then single columns iteratively."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        removed = 0
        
        def try_remove(cols_to_remove, Ua, Va, Wa):
            """Check if removing given columns still gives valid decomposition."""
            keep = [c for c in range(Ua.shape[1]) if c not in cols_to_remove]
            if not keep:
                return None, None, None
            Uo = Ua[:, keep]
            Vo = Va[:, keep]
            Wo = Wa[:, keep]
            # Quick verification with a few random inputs
            rng = np.random.RandomState(42 + hash(tuple(sorted(cols_to_remove))) % 10000)
            for _ in range(3):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n*m) @ Uo)
                y = (B.reshape(m*p) @ Vo)
                prod = x * y
                C_computed = (Wo * prod[None, :]).sum(axis=1).reshape(p, n)
                C_exp_T = C_expected.T
                if not np.allclose(C_computed, C_exp_T, atol=1e-7, rtol=1e-7):
                    return None, None, None
            return Uo, Vo, Wo
        
        # Phase 1: Try removing pairs of columns (most likely to find dependencies)
        for _ in range(20):  # Limit iterations
            if elapsed() > DEADLINE - 20:
                break
            if rank <= 2:
                break
            
            found_pair = False
            # Sample a subset of pairs to try (not all O(r^2))
            rng = np.random.RandomState(int(elapsed() * 1000) % 100000)
            pairs_to_try = []
            for _ in range(min(200, rank * (rank-1) // 2)):
                i, j = rng.choice(rank, 2, replace=False)
                pairs_to_try.append((min(i,j), max(i,j)))
            pairs_to_try = list(set(pairs_to_try))
            
            for i, j in pairs_to_try:
                if elapsed() > DEADLINE - 20:
                    break
                Uo, Vo, Wo = try_remove((i, j), Ua, Va, Wa)
                if Uo is not None:
                    Ua, Va, Wa = Uo, Vo, Wo
                    rank -= 2
                    removed += 2
                    found_pair = True
                    break
            
            if not found_pair:
                break
        
        # Phase 2: Try removing single columns iteratively
        for _ in range(30):
            if elapsed() > DEADLINE - 20:
                break
            if rank <= 1:
                break
            
            found_single = False
            for c in range(rank):
                if elapsed() > DEADLINE - 20:
                    break
                Uo, Vo, Wo = try_remove((c,), Ua, Va, Wa)
                if Uo is not None:
                    Ua, Va, Wa = Uo, Vo, Wo
                    rank -= 1
                    removed += 1
                    found_single = True
                    break
            
            if not found_single:
                break
        
        # Phase 3: Try triples
        for _ in range(10):
            if elapsed() > DEADLINE - 20:
                break
            if rank <= 3:
                break
            
            found_triple = False
            rng = np.random.RandomState(int(elapsed() * 2000) % 100000)
            triples_to_try = []
            for _ in range(min(100, rank * (rank-1) * (rank-2) // 6)):
                idx = rng.choice(rank, 3, replace=False)
                triples_to_try.append(tuple(sorted(idx)))
            triples_to_try = list(set(triples_to_try))
            
            for triple in triples_to_try:
                if elapsed() > DEADLINE - 20:
                    break
                Uo, Vo, Wo = try_remove(triple, Ua, Va, Wa)
                if Uo is not None:
                    Ua, Va, Wa = Uo, Vo, Wo
                    rank -= 3
                    removed += 3
                    found_triple = True
                    break
            
            if not found_triple:
                break
        
        def to_list(a):
            return [[(float(v)) for v in row] for row in a]
        return to_list(Ua), to_list(Va), to_list(Wa), removed

    def linear_dependency_reduce(U, V, W, max_iter=5):
        """Use linear algebra to find exact linear dependencies among terms.
        For each column c, check if its contribution tensor lies in the span
        of other columns' contribution tensors."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        removed = 0
        
        def tensor_signature(Ua, Va, Wa, col):
            """Get the flattened tensor signature of column col."""
            ua = Ua[:, col]
            va = Va[:, col]
            wa = Wa[:, col]
            # Tensor T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
            # Flatten into a vector
            sig = np.zeros(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sig[k*n + i, ] = 0  # placeholder
            # More efficient: use einsum
            T = np.einsum('a,b,c->abc', ua, va, wa)
            # But indices are: U indexed by (i,j), V by (j,k), W by (k,i)
            # So T[i,j,k] = U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
            sig = np.zeros(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sig[(i*m+j)*p + k] = ua[i*m+j] * va[j*p+k] * wa[k*n+i]
            return sig
        
        for iteration in range(max_iter):
            if elapsed() > DEADLINE - 25:
                break
            if rank <= 1:
                break
            
            # Compute signatures for all columns
            sigs = np.array([tensor_signature(Ua, Va, Wa, c) for c in range(rank)])
            # Check if any column's signature is in the span of others
            found = False
            for c in range(rank):
                if elapsed() > DEADLINE - 25:
                    break
                # Check if sig[c] is in span of all other sigs
                others = np.delete(sigs, c, axis=0)
                # Compute residual: sig[c] - projection onto span(others)
                # Use least squares
                if others.shape[0] >= rank - 1:
                    try:
                        coeffs, residuals, _, _ = np.linalg.lstsq(others, sigs[c], rcond=None)
                        residual = sigs[c] - others @ coeffs
                        if np.linalg.norm(residual) < 1e-8 * max(1, np.linalg.norm(sigs[c])):
                            # Column c is in span of others; try removing it
                            Uo = np.delete(Ua, c, axis=1)
                            Vo = np.delete(Va, c, axis=1)
                            Wo = np.delete(Wa, c, axis=1)
                            # Verify
                            rng = np.random.RandomState(999 + c + iteration*100)
                            ok = True
                            for _ in range(5):
                                A = rng.rand(n, m)
                                B = rng.rand(m, p)
                                C_expected = A @ B
                                x = (A.reshape(n*m) @ Uo)
                                y = (B.reshape(m*p) @ Vo)
                                prod = x * y
                                C_computed = (Wo * prod[None, :]).sum(axis=1).reshape(p, n)
                                C_exp_T = C_expected.T
                                if not np.allclose(C_computed, C_exp_T, atol=1e-7, rtol=1e-7):
                                    ok = False
                                    break
                            if ok:
                                Ua, Va, Wa = Uo, Vo, Wo
                                rank -= 1
                                removed += 1
                                found = True
                                break
                    except Exception:
                        pass
            
            if not found:
                break
        
        def to_list(a):
            return [[(float(v)) for v in row] for row in a]
        return to_list(Ua), to_list(Va), to_list(Wa), removed

    # ---- Build the Kronecker product decomposition: Strassen(2x2x2) ⊗ (2x2x5) ----
    # The 2x2x5 factor: for cols 0-3, use Strassen-like; for col 4, use identity
    # Actually, let's construct it directly:
    # 2x2x5 tensor: T[i,j,k] = delta(i,j,k) for i in {0,1}, j in {0,1}, k in {0,...,4}
    # This is just the identity mapping: T[i,j,k] = 1 if j==k and k<2, else 0... 
    # Wait, for 2x2x5: A is 2x2, B is 2x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the standard bilinear form)
    # Rank of 2x2x5 = 2*2*5 = 20 (schoolbook) or can we do better?
    # Actually rank(2x2x5) >= 5 (since it contains 2x2x2 as a sub-tensor, rank >= 7)
    # But 2x2x5 is just 5 copies of 2x2x1 (each column of B)
    # Each 2x2x1 has rank 4 (schoolbook) or 3 (Strassen doesn't help for 1 output)
    # So 2x2x5 has rank at most 5*4 = 20, and at least... 
    # Actually, the 2x2x5 tensor is the direct sum of 5 copies of the 2x2x1 tensor
    # The 2x2x1 tensor (single output element) has rank 4 (it's just the identity map R^2 x R^2 -> R, no wait)
    # 2x2x1: C = A*B where A is 2x2, B is 2x1, C is 2x1
    # C[i] = sum_j A[i,j] B[j]
    # T[i,j,0] = delta(i,j) ... no, T[i,j,k] with k=0 only
    # T[i,j,0] = 1 if we're computing C[i] = sum_j A[i,j]*B[j]
    # So T[i,j,0] = 1 for all i,j (the coefficient of A[i,j]*B[j] in C[i] is 1)
    # Wait no: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # So T[i,0,0] = 1, T[i,1,0] = 1 for i=0,1
    # That's a 2x2x1 tensor with all entries = 1, which has rank 1!
    # T = ones(2,2,1) = u ⊗ v ⊗ w where u=ones(2), v=ones(2), w=[1]
    # So 2x2x1 has rank 1!
    # Therefore 2x2x5 = direct sum of 5 copies of 2x2x1, each rank 1, total rank 5.
    
    # So the Kronecker product: Strassen(2x2x2, rank 7) ⊗ (2x2x5, rank 5) = rank 35
    # But wait, the Kronecker product of tensors: if T has rank r and S has rank s,
    # then T ⊗ S has rank at most r*s, and exactly r*s if both are "generic" enough.
    # Actually, rank(T ⊗ S) can be less than rank(T)*rank(S) in some cases.
    
    # Let me construct this explicitly.
    # Strassen 2x2x2: 7 terms
    # 2x2x5: 5 terms (one for each output column)
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    strassen_U = [
        # Term 0: M1 = (A00+A11)*(B00+B11)
        [[1,0],[1,0]],  # A00+A11: rows are (i,j) pairs
        # Actually let me be more careful
    ]
    
    # Let me define Strassen 2x2x2 explicitly:
    # U factors: for each term, which (i,j) entries of A are used and with what coefficient
    # V factors: for each term, which (j,k) entries of B are used
    # W factors: for each term, which (k,i) entries of C are produced
    
    # Term 0: M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    # U: A00=1, A11=1 => [(0,0,1), (1,1,1)]
    # V: B00=1, B11=1 => [(0,0,1), (1,1,1)]
    # W: C00=1, C11=1 => [(0,0,1), (1,1,1)]
    
    # Term 1: M2 = (A10+A11)*B00 -> C10 += 1, C11 -= 1
    # U: [(1,0,1), (1,1,1)]
    # V: [(0,0,1)]
    # W: [(0,1,1), (1,1,-1)]
    
    # Term 2: M3 = A00*(B10-B11) -> C01 += 1, C11 += 1
    # U: [(0,0,1)]
    # V: [(1,0,1), (1,1,-1)]
    # W: [(1,0,1), (1,1,1)]
    
    # Term 3: M4 = A11*(B01-B10) -> C00 += 1, C10 += 1
    # U: [(1,1,1)]
    # V: [(0,1,1), (1,0,-1)]
    # W: [(0,0,1), (0,1,1)]
    
    # Term 4: M5 = (A00+A01)*B11 -> C00 -= 1, C01 += 1
    # U: [(0,0,1), (0,1,1)]
    # V: [(1,1,1)]
    # W: [(0,0,-1), (1,0,1)]
    
    # Term 5: M6 = (A10-A00)*(B00+B01) -> C11 += 1
    # U: [(1,0,1), (0,0,-1)]
    # V: [(0,0,1), (0,1,1)]
    # W: [(1,1,1)]
    
    # Term 6: M7 = (A01-A11)*(B10+B11) -> C00 += 1
    # U: [(0,1,1), (1,1,-1)]
    # V: [(1,0,1), (1,1,1)]
    # W: [(0,0,1)]
    
    strassen_terms = [
        # (U_list, V_list, W_list) where each is list of (index1, index2, coeff)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),           # M2
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1),(1,1,1)]),           # M3
        ([(1,1,1)], [(0,1,1),(1,0,-1)], [(0,0,1),(0,1,1)]),           # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),           # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),           # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),           # M7
    ]
    
    # 2x2x5 decomposition (rank 5):
    # For each output column k in 0..4:
    # C[:,k] = A * B[:,k]
    # This is a 2x2x1 problem for each k, which has rank 1:
    # T[i,j,k] = 1 for all i,j (since C[i,k] = sum_j A[i,j]*B[j,k])
    # So U_factor for term k: A[i,j] with coeff 1 for all i,j
    # V_factor for term k: B[j,k] with coeff 1
    # W_factor for term k: C[i,k] with coeff 1
    
    # Wait, that's not right. Let me reconsider.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = delta(i,j) ... no.
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    # But that's a rank-1 tensor! T = ones(2,2,5) = u ⊗ v ⊗ w
    # where u[i]=1, v[j]=1, w[k]=1.
    # So the entire 2x2x5 tensor has rank 1?!
    # No wait, that can't be right because then 2x2x5 would have rank 1,
    # but we know matrix multiplication 2x2x2 has rank at least 7.
    # 
    # I'm confusing myself. Let me be very careful.
    # The matrix multiplication tensor for 2x2x5:
    # Inputs: A (2x2), B (2x5)
    # Output: C (2x5)
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # 
    # The tensor T is a trilinear form: T(A,B,C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For the multiplication to work, we need:
    # sum_j A[i,j]*B[j,k] = C[i,k] for all i,k
    # So T[i,j,k] should be such that the bilinear form sum_{j} T[i,j,k]*A[i,j]*B[j,k] = C[i,k]
    # This means T[i,j,k] = 1 for all i,j,k? No...
    # Actually, the standard way: the tensor T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Since C[i,k] = sum_j A[i,j]*B[j,k], every (i,j,k) triple with the constraint that j is the shared index
    # contributes. So T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,4}.
    # 
    # But this tensor is NOT rank 1. The rank-1 decomposition would be:
    # T = sum_alpha u_alpha ⊗ v_alpha ⊗ w_alpha
    # where u_alpha[i,j] * v_alpha[j,k] * w_alpha[i,k] sums to T[i,j,k]
    # 
    # For T[i,j,k] = 1 for all i,j,k, we need:
    # sum_alpha u_alpha[i,j] * v_alpha[j,k] * w_alpha[i,k] = 1
    # 
    # This is the tensor of all ones of shape 2x2x5. Its rank is... 
    # The all-ones tensor of shape a×b×c has rank min(a,b,c) if... no.
    # Actually, the all-ones tensor J[a,b,c] has rank = min(a,b,c) when one dimension is 1?
    # No, J[2,2,2] has rank 2 (it's the "sum" tensor).
    # J[2,2,5]: rank is at most 4 (min(2*2, 2*5, 2*5) = 4)? 
    # Actually, I think the rank of the all-ones tensor J[a,b,c] is min(a,b,c) when... 
    # Let me just use the standard result: the all-ones tensor has rank = min(a,b,c) only in specific cases.
    # 
    # I'm overcomplicating this. Let me just use the schoolbook decomposition for 2x2x5,
    # which has rank 2*2*5 = 20, and then the Kronecker product with Strassen gives 7*20 = 140.
    # That's worse than 65!
    # 
    # The correct approach: the 4x4x5 tensor is NOT simply Strassen(2x2x2) ⊗ (2x2x5).
    # The Kronecker product structure works when the multiplication can be written as a block product
    # where each block uses the same pattern.
    # 
    # For 4x4x5 = (2x2 ⊗ 2x2) x (2x2 ⊗ 2x2.5)... this doesn't work cleanly.
    # 
    # Let me go back to the original approach: double-Strassen for 4x4x4 + schoolbook for 4x4x1.
    # But let me improve the construction to be cleaner and add better rank reduction.
    
    # Actually, let me reconsider the Kronecker approach more carefully.
    # 4x4x5: A is 4x4, B is 4x5, C is 4x5
    # We can view A as a 2x2 block matrix of 2x2 blocks, B as a 2x2 block matrix of 2x2.5 blocks... no.
    # 
    # Alternative: A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x... no, B is 4x5.
    # 
    # Let's partition differently:
    # A is 4x4: partition into 2x2 blocks of size 2x2
    # B is 4x5: partition rows into 2x2 blocks (4 rows = 2 blocks of 2), columns into... 5 doesn't divide evenly.
    # 
    # So the clean Kronecker approach doesn't work for 4x4x5 because 5 is not a multiple of 2.
    # 
    # The best we can do is:
    # - 4x4x4 part: use double-Strassen (rank 49)
    # - 4x4x1 part: use schoolbook (rank 16)
    # Total: 65
    # 
    # OR:
    # - 4x4x4 part: use Strassen(2x2x2) ⊗ Strassen(2x2x2) = rank 49
    # - 4x4x1 part: use Strassen(2x2x2) ⊗ schoolbook(2x2x1) = rank 7*4 = 28? 
    #   Wait, 2x2x1 has rank 4 (schoolbook). Strassen doesn't help for 1 output.
    #   Actually, 2x2x1: C = A*B where A is 2x2, B is 2x1, C is 2x1
    #   C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #   C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #   This is just the identity map applied to each row of A times B.
    #   Rank of 2x2x1 = 4 (schoolbook is optimal for 1 output column).
    #   So 4x4x1 via Kronecker: Strassen(2x2x2) ⊗ schoolbook(2x2x1) = 7*4 = 28? 
    #   But that's worse than schoolbook 4x4x1 = 16.
    # 
    # So the optimal split is: 49 + 16 = 65.
    # 
    # Let me focus on improving the rank reduction from 65.
    
    # Build double-Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = 65
    
    rank = 65
    U = np.zeros((n*m, rank))  # 16 x 65
    V = np.zeros((m*p, rank))  # 20 x 65
    W = np.zeros((p*n, rank))  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i*m + j, col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j*p + k, col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k*n + i, col] += c
    
    col = 0
    
    # Inner Strassen 2x2x2 terms (for the 2x2 blocks)
    inner_defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),               # M2
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1),(1,1,1)]),               # M3
        ([(1,1,1)], [(0,1,1),(1,0,-1)], [(0,0,1),(0,1,1)]),               # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),               # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),               # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),               # M7
    ]
    
    # Super Strassen 4x4x4: 7 outer terms
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7
    ]
    
    def get_p_element(p_elem_idx, p_block_coeffs):
        """Map 2x2 block element index to global 4x4 indices."""
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_element(q_elem_idx, q_block_coeffs):
        """Map 2x2 block element index to global 4x5 indices."""
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16) for the 5th column of B
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
    
    # Verify the baseline
    baseline_ok = verify(U, V, W, seed=99, trials=10)
    if not baseline_ok:
        # Fallback: return schoolbook
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i*m+j)*p + k
                    U_sb[i*m+j, idx] = 1
                    V_sb[j*p+k, idx] = 1
                    W_sb[k*n+i, idx] = 1
        def to_list(a):
            return [[(float(v)) for v in row] for row in a]
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Attempt rank reduction with multiple strategies
    if elapsed() < DEADLINE - 40:
        # Strategy 1: Exhaustive pair/triple removal
        Ur, Vr, Wr, removed = exhaustive_rank_reduce(U, V, W, max_removed=10)
        if removed > 0 and verify(Ur, Vr, Wr, seed=777, trials=15):
            best_U, best_V, best_W = Ur, Vr, Wr
            best_rank = len(Ur[0]) if Ur else 0
        
        if elapsed() < DEADLINE - 30:
            # Strategy 2: Linear dependency search
            Ur2, Vr2, Wr2, removed2 = linear_dependency_reduce(best_U, best_V, best_W, max_iter=3)
            if removed2 > 0 and verify(Ur2, Vr2, Wr2, seed=888, trials=15):
                best_U, best_V, best_W = Ur2, Vr2, Wr2
                best_rank = len(Ur2[0]) if Ur2 else 0
        
        if elapsed() < DEADLINE - 20:
            # Strategy 3: Another round of exhaustive removal
            Ur3, Vr3, Wr3, removed3 = exhaustive_rank_reduce(best_U, best_V, best_W, max_removed=5)
            if removed3 > 0 and verify(Ur3, Vr3, Wr3, seed=999, trials=15):
                best_U, best_V, best_W = Ur3, Vr3, Wr3
                best_rank = len(Ur3[0]) if Ur3 else 0
    
    def to_list(a):
        if isinstance(a, np.ndarray):
            return [[(float(v)) for v in row] for row in a]
        return a
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
