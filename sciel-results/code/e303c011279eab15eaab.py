import numpy as np
import time
import itertools
import random

# Global deadline (120s wall-clock, leave 10s safety margin)
_START_TIME = time.time()
_DEADLINE = 110.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m      # 6
    N_V = m * p      # 21
    N_W = p * n      # 14

    # ---------- Build the target tensor T of shape (N_U, N_V, N_W) ----------
    # Indexing: U: i*m+j, V: j*p+k, W: k*n+i
    T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    # ---------- Verify helper ----------
    def verify(U, V, W):
        """U: (N_U, R), V: (N_V, R), W: (N_W, R) -> bool"""
        T_calc = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_calc, T_target, atol=1e-8, rtol=1e-8)

    # ---------- Border rank lower bound ----------
    def border_rank_lb(r_test):
        """
        Returns True if border rank > r_test (i.e., the degree-(r_test+1)
        tensor does NOT vanish on X^n x Y^m x Z^p).
        """
        d = r_test + 1
        # Collect all monomials x^a y^b z^c with a+b+c = d
        monos = []
        for a in range(d+1):
            for b in range(d+1 - a):
                c = d - a - b
                # Multinomial coefficient
                import math
                coeff = math.comb(d, a) * math.comb(d - a, b)
                monos.append((a, b, c, coeff))

        # We need to check if the polynomial
        #   F(x,y,z) = sum_{a+b+c=d} C(a,b,c) x^a y^b z^c
        # vanishes for all x in X^n, y in Y^m, z in Z^p
        # where X^n = { u v^T : u in R^n, v in R^n } etc.
        #
        # A monomial x^a y^b z^c on X^n x Y^m x Z^p:
        #   x = u v^T (n x n), so x_{ii'} = u_i v_{i'}
        #   y = u' v'^T (m x m), so y_{jj'} = u'_j v'_{j'}
        #   z = u'' v''^T (p x p), so z_{kk'} = u''_k v''_{k'}
        #
        # The tensor T(a,b,c)_{i,j,k, i',j',k'} = T_target expanded:
        #   T(a,b,c) = sum over indices... 
        #
        # Actually, the standard test: form the matrix M of shape
        #   (n^2 * m^2 * p^2) x (number of monomials)
        # where row (i,i',j,j',k,k') and column (a,b,c) is
        #   <T, x^{(a)} y^{(b)} z^{(c)}> evaluated at (x,y,z) = (u v^T, u' v'^T, u'' v''^T)
        #
        # For the matrix multiplication tensor, the contraction is:
        #   <T, x y z> = sum_{i,j,k} x_{i,j} y_{j,k} z_{k,i}
        #
        # So for x = u v^T, y = u' v'^T, z = u'' v''^T:
        #   sum_{i,j,k} (u_i v_{i'})(u'_j v'_{j'})(u''_k v''_{k'}) * delta_{j, (i,j) col} ...
        #
        # Wait, I need to be more careful. The tensor T has indices (a,b,c) = (i*m+j, j*p+k, k*n+i).
        # The bilinear form <T, X> for X of shape (N_U, N_V, N_W) is sum_{a,b,c} T_{a,b,c} X_{a,b,c}.
        #
        # For X_{a,b,c} = x_{a} y_{b} z_{c} where x has shape (N_U,), y shape (N_V,), z shape (N_W,):
        #   <T, x y z> = sum_{i,j,k} x_{i*m+j} y_{j*p+k} z_{k*n+i}
        #
        # Now if x = vec(X_mat) where X_mat = u v^T (n x n reshaped), etc.
        # This is getting complicated. Let me use a different approach.
        #
        # Alternative: The border rank test via the "catalecticant" or "flattening".
        # For the matrix multiplication tensor M_{n,m,p}, the border rank lower bound
        # can be computed by checking the rank of certain flattening matrices.
        #
        # Actually, the simplest correct test: 
        #   M_{n,m,p} has border rank > r iff the polynomial
        #   P(x, y, z) = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1}
        #       x_{i,j} y_{j,k} z_{k,i}
        #   does NOT vanish on the product of the varieties of rank-1 matrices.
        #
        # For the degree-d test: we need to check if the homogeneous polynomial of degree d
        # in the entries of X (n x n), Y (m x m), Z (p x p)
        #   (the d-th power of the tensor, symmetrized)
        # vanishes on rank-1 matrices.
        #
        # This is equivalent to: the matrix
        #   B of shape (n^2 * m^2 * p^2) x (dim of degree-d polynomials vanishing...)
        # has rank less than the number of monomials.
        #
        # Let me just use a simpler sufficient condition.
        #
        # For our purposes, I'll use the known lower bound: 
        #   R(n,m,p) >= n*m*p / max(n,m,p) is NOT a valid lower bound.
        #   The correct simple lower bound: R(n,m,p) >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
        #
        # And R(n,m,p) >= n + m + p - 2 + ... (no simple formula).
        #
        # Let me just use R >= 21 as a trivial lower bound and focus on search.
        return False  # Skip the complex test, use trivial bound

    # Trivial lower bound
    lb = max(n*m, m*p, p*n)  # = 21

    # ---------- Known rank-39 fallback ----------
    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        U_s = np.zeros((4, 7)); V_s = np.zeros((4, 7)); W_s = np.zeros((4, 7))
        U_s[0,0]=1; U_s[3,0]=1; V_s[0,0]=1; V_s[3,0]=1; W_s[0,0]=1; W_s[3,0]=1
        U_s[2,1]=1; U_s[3,1]=1; V_s[0,1]=1; W_s[2,1]=1; W_s[3,1]=-1
        U_s[0,2]=1; V_s[1,2]=1; V_s[3,2]=-1; W_s[1,2]=1; W_s[3,2]=1
        U_s[3,3]=1; V_s[2,3]=1; V_s[0,3]=-1; W_s[0,3]=1; W_s[2,3]=1
        U_s[0,4]=1; U_s[1,4]=1; V_s[3,4]=1; W_s[0,4]=-1; W_s[1,4]=1
        U_s[2,5]=1; U_s[0,5]=-1; V_s[0,5]=1; V_s[1,5]=1; W_s[3,5]=1
        U_s[1,6]=1; U_s[3,6]=-1; V_s[2,6]=1; V_s[3,6]=1; W_s[0,6]=1
        U[0,0:7]=U_s[0,:]; U[1,0:7]=U_s[1,:]; U[3,0:7]=U_s[2,:]; U[4,0:7]=U_s[3,:]
        V[0,0:7]=V_s[0,:]; V[1,0:7]=V_s[1,:]; V[2,0:7]=V_s[2,:]; V[3,0:7]=V_s[3,:]
        W[0,0:7]=W_s[0,:]; W[1,0:7]=W_s[1,:]; W[2,0:7]=W_s[2,:]; W[3,0:7]=W_s[3,:]
        U_212=np.zeros((2,4)); V_212=np.zeros((2,4)); W_212=np.zeros((4,4))
        U_212[0,0]=1; V_212[0,0]=1; W_212[0,0]=1
        U_212[0,1]=1; V_212[1,1]=1; W_212[1,1]=1
        U_212[1,2]=1; V_212[0,2]=1; W_212[2,2]=1
        U_212[1,3]=1; V_212[1,3]=1; W_212[3,3]=1
        U[2,7:11]=U_212[0,:]; U[5,7:11]=U_212[1,:]
        V[4,7:11]=V_212[0,:]; V[5,7:11]=V_212[1,:]
        W[0,7:11]=W_212[0,:]; W[1,7:11]=W_212[1,:]; W[2,7:11]=W_212[2,:]; W[3,7:11]=W_212[3,:]
        return U, V, W

    def get_2x3x1_rank6():
        U=np.zeros((6,6)); V=np.zeros((3,6)); W=np.zeros((2,6))
        col=0
        for i in range(2):
            for j in range(3):
                U[i*3+j,col]=1; V[j,col]=1; W[i,col]=1; col+=1
        return U,V,W

    def build_2x3x7_rank39():
        U11a,V11a,W11a=get_2x3x2_rank11()
        U11b,V11b,W11b=get_2x3x2_rank11()
        U11c,V11c,W11c=get_2x3x2_rank11()
        U6,V6,W6=get_2x3x1_rank6()
        R=39
        Uf=np.zeros((N_U,R)); Vf=np.zeros((N_V,R)); Wf=np.zeros((N_W,R))
        Uf[:,0:11]=U11a
        for j in range(3):
            for k in range(2):
                Vf[j*7+k,0:11]=V11a[j*2+k,:]
        for k in range(2):
            for i in range(2):
                Wf[k*2+i,0:11]=W11a[k*2+i,:]
        Uf[:,11:22]=U11b
        for j in range(3):
            for k in range(2):
                Vf[j*7+(k+2),11:22]=V11b[j*2+k,:]
        for k in range(2):
            for i in range(2):
                Wf[(k+2)*2+i,11:22]=W11b[k*2+i,:]
        Uf[:,22:33]=U11c
        for j in range(3):
            for k in range(2):
                Vf[j*7+(k+4),22:33]=V11c[j*2+k,:]
        for k in range(2):
            for i in range(2):
                Wf[(k+4)*2+i,22:33]=W11c[k*2+i,:]
        Uf[:,33:39]=U6
        for j in range(3):
            Vf[j*7+6,33:39]=V6[j,:]
        for i in range(2):
            Wf[6*2+i,33:39]=W6[i,:]
        return Uf, Vf, Wf

    def get_schoolbook():
        rank = n*m*p
        U=np.zeros((N_U,rank)); V=np.zeros((N_V,rank)); W=np.zeros((N_W,rank))
        col=0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j,col]=1; V[j*p+k,col]=1; W[k*n+i,col]=1; col+=1
        return {"U":U.tolist(),"V":V.tolist(),"W":W.tolist()}

    # ---------- Randomized ALS search ----------
    def random_als_search(rank, max_iters=200, seed=42):
        """
        Randomized alternating least squares to find a rank-`rank` decomposition.
        Returns (U, V, W) or None.
        """
        rng = np.random.RandomState(seed)
        # Initialize with random rank-`rank` decomposition
        U = rng.randn(N_U, rank)
        V = rng.randn(N_V, rank)
        W = rng.randn(N_W, rank)

        # Precompute target tensor as a 3D array
        # T_{a,b,c} = 1 if (a,b,c) corresponds to valid (i,j,k)
        # We'll use einsum to compute the error

        for it in range(max_iters):
            if _time_left() < 15.0:
                return None

            # Update U: for each r, U[:,r] = argmin sum_{a,b,c} (T_{a,b,c} - sum_r' U_{a,r'}V_{b,r'}W_{c,r'})^2 w.r.t. U_{a,r}
            # This is a linear system: 
            #   sum_b,c T_{a,b,c} V_{b,r} W_{c,r} ... no, this is not linear in U.
            # 
            # Actually, ALS for tensor decomposition:
            #   Fix V and W. The problem becomes: minimize ||T - U * V * W||_F^2 w.r.t. U.
            #   T_{a,b,c} - sum_r U_{a,r} V_{b,r} W_{c,r}
            #   = T_{a,b,c} - (U V W)_{a,b,c}
            #   where (U V W)_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            #
            # This is NOT a simple linear system in U because V and W are also variables.
            # But if we fix V and W, it IS linear in U:
            #   min_U sum_{a,b,c} (T_{a,b,c} - sum_r U_{a,r} V_{b,r} W_{c,r})^2
            #   = min_U ||T - U (V^T * W^T)||^2  (where * is the Kronecker product)
            #
            # More precisely, the (b,c) slice of the "effective target" for U is:
            #   For each a: T_{a,:} - sum_r U_{a,r} (V[:,r] * W[:,r])
            #   where (V[:,r] * W[:,r])_b = V_{b,r} W_{b,r} ... no, that's element-wise.
            #
            # Let me think again. We want:
            #   sum_r U_{a,r} V_{b,r} W_{c,r} ≈ T_{a,b,c}
            #   For fixed V, W and a given r, the contribution to (a,b,c) is U_{a,r} * (V_{b,r} W_{c,r}).
            #   So for fixed V, W, the problem is:
            #   min_{U} sum_{a,b,c} (T_{a,b,c} - sum_r U_{a,r} M_{b,c,r})^2
            #   where M_{b,c,r} = V_{b,r} W_{c,r}
            #
            # This is a linear least squares problem for each a:
            #   min_{U_{a,:}} ||t_a - sum_r U_{a,r} m_r||^2
            #   where t_a ∈ R^{N_V * N_W} is the a-th slice of T, and m_r ∈ R^{N_V * N_W}
            #   is the vectorized M[:, :, r].
            #
            # So: U[a, :] = (M M^T)^{-1} M t_a  (ridge: (M M^T + λI)^{-1} M t_a)
            #
            # Similarly for V and W.

            lambda_reg = 1e-6
            # Build M = V * W (element-wise, shape N_V x N_W x rank)
            M = V * W[np.newaxis, :, :]  # (N_V, N_W, rank)
            M_flat = M.reshape(N_V * N_W, rank)  # (N_V*N_W, rank)

            # For each a, solve: min_U_a ||T[a,:,:].flat - M_flat @ U_a||^2
            T_flat = T_target.reshape(N_U, N_V * N_W)  # (N_U, N_V*N_W)
            # U = (M_flat^T M_flat + λI)^{-1} M_flat^T T_flat^T
            MTM = M_flat.T @ M_flat  # (rank, rank)
            MTM += lambda_reg * np.eye(rank)
            U_new = np.linalg.solve(MTM, M_flat.T @ T_flat.T).T  # (N_U, rank)

            # Update V
            MW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (N_U, N_W, rank)
            MW_flat = MW.reshape(N_U * N_W, rank)
            T_flat_v = T_target.reshape(N_U * N_W, N_V).T  # (N_V, N_U*N_W)
            MTM_v = MW_flat.T @ MW_flat + lambda_reg * np.eye(rank)
            V_new = np.linalg.solve(MTM_v, MW_flat.T @ T_flat_v.T).T  # (N_V, rank)

            # Update W
            MV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (N_U, N_V, rank)
            MV_flat = MV.reshape(N_U * N_V, rank)
            T_flat_w = T_target.reshape(N_U * N_V, N_W).T  # (N_W, N_U*N_V)
            MTM_w = MV_flat.T @ MV_flat + lambda_reg * np.eye(rank)
            W_new = np.linalg.solve(MTM_w, MV_flat.T @ T_flat_w.T).T  # (N_W, rank)

            U, V, W = U_new, V_new, W_new

            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T_calc - T_target)
            if err < 1e-6:
                return U, V, W

        # Final check
        T_calc = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if np.allclose(T_calc, T_target, atol=1e-5, rtol=1e-5):
            return U, V, W
        return None

    # ---------- Search strategy ----------
    best_rank = 39
    best_decomp = None

    # Try ranks from 18 up to 24 (below 39)
    for rank in range(18, 25):
        if _time_left() < 20.0:
            break
        # Try multiple random seeds
        for seed in range(20):
            if _time_left() < 15.0:
                break
            result = random_als_search(rank, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_rank = rank
                    best_decomp = (U, V, W)
                    break  # Found a valid decomposition at this rank, try lower
        if best_rank < 39:
            break

    # If we found a better decomposition, return it
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: return the known rank-39 decomposition
    Uf, Vf, Wf = build_2x3x7_rank39()
    if verify(Uf, Vf, Wf):
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # Ultimate fallback: schoolbook
    return get_schoolbook()
