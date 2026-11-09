# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

class _LocalSearcher:
    def __init__(self, n, m, p, r, U, V, W):
        self.n, self.m, self.p, self.r = n, m, p, r
        self.U = U
        self.V = V
        self.W = W
        # Precompute T and Error
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        self.T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
        self.err = np.sum((self.T - 1.0)**2)

    def update_U(self, i, j, c, new_val):
        # Update U[i*m+j, c]
        idx = i * self.m + j
        old_val = self.U[idx, c]
        if old_val == new_val:
            return False
        self.U[idx, c] = new_val
        delta = new_val - old_val
        # T[i, :, :] += delta * V[:, c] * W[:, c] ?
        # T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Change in T[i, j, k] = delta * V[j, k, c] * W[k, i, c]
        # V is (m, p, r) -> V[j, :, c] is vector of size p
        # W is (p, n, r) -> W[:, i, c] is vector of size p
        # So update T[i, j, k] for all j, k
        # V_jk = V[j, k, c]
        # W_ki = W[k, i, c]
        # Update T[i, j, k] += delta * V[j, k, c] * W[k, i, c]
        
        # Reshape V and W for easier access
        # V is (m, p, r). V[j, :, c] is (p,)
        # W is (p, n, r). W[:, i, c] is (p,)
        
        v_vec = self.V[j, :, c] # size p
        w_vec = self.W[:, i, c] # size p
        
        # Outer product v_vec * w_vec.T gives (p, p)
        # We need to update T[i, j, k] where j is fixed (current j), k varies?
        # Wait, the index j in U is the middle dimension of the first matrix.
        # In T[i, j, k], j is the column of A (row of B).
        # So for a fixed i and j (from U), we update T[i, j, k] for all k.
        # The term is V[j, k, c] * W[k, i, c].
        # So for each k, T[i, j, k] += delta * V[j, k, c] * W[k, i, c].
        
        # V[j, :, c] is (p,). W[:, i, c] is (p,).
        # Element k is V[j, k, c] * W[k, i, c].
        # This is element-wise product of v_vec and w_vec.
        
        # Update T[i, j, :] += delta * (v_vec * w_vec)
        
        old_T_slice = self.T[i, j, :]
        new_T_slice = old_T_slice + delta * (v_vec * w_vec)
        
        # Update error
        old_err_slice = np.sum((old_T_slice - 1.0)**2)
        new_err_slice = np.sum((new_T_slice - 1.0)**2)
        
        self.T[i, j, :] = new_T_slice
        self.err += new_err_slice - old_err_slice
        return True

    def update_V(self, j, k, c, new_val):
        idx = j * self.p + k
        old_val = self.V[idx, c]
        if old_val == new_val:
            return False
        self.V[idx, c] = new_val
        delta = new_val - old_val
        # T[i, j, k] += delta * U[i, j, c] * W[k, i, c]
        # For fixed j, k, we update T[i, j, k] for all i.
        # u_vec = U[:, j, c] (size n)
        # w_val = W[k, i, c] (depends on i)
        # T[i, j, k] += delta * U[i, j, c] * W[k, i, c]
        
        u_vec = self.U[:, j, c] # size n
        # w_vec for fixed k: W[k, :, c] is size n
        w_vec = self.W[k, :, c] # size n
        
        # Element i is U[i, j, c] * W[k, i, c]
        # Update T[:, j, k] += delta * (u_vec * w_vec)
        
        old_T_slice = self.T[:, j, k]
        new_T_slice = old_T_slice + delta * (u_vec * w_vec)
        
        old_err_slice = np.sum((old_T_slice - 1.0)**2)
        new_err_slice = np.sum((new_T_slice - 1.0)**2)
        
        self.T[:, j, k] = new_T_slice
        self.err += new_err_slice - old_err_slice
        return True

    def update_W(self, k, i, c, new_val):
        idx = k * self.n + i
        old_val = self.W[idx, c]
        if old_val == new_val:
            return False
        self.W[idx, c] = new_val
        delta = new_val - old_val
        # T[i, j, k] += delta * U[i, j, c] * V[j, k, c]
        # For fixed k, i, we update T[i, j, k] for all j.
        # u_val = U[i, j, c] (depends on j)
        # v_val = V[j, k, c] (depends on j)
        
        # u_vec = U[:, i, c] ? No, U is (n, m, r). U[i, :, c] is size m.
        # v_vec = V[:, k, c] ? No, V is (m, p, r). V[:, k, c] is size m.
        
        u_vec = self.U[i, :, c] # size m
        v_vec = self.V[:, k, c] # size m
        
        # Element j is U[i, j, c] * V[j, k, c]
        # Update T[i, :, k] += delta * (u_vec * v_vec)
        
        old_T_slice = self.T[i, :, k]
        new_T_slice = old_T_slice + delta * (u_vec * v_vec)
        
        old_err_slice = np.sum((old_T_slice - 1.0)**2)
        new_err_slice = np.sum((new_T_slice - 1.0)**2)
        
        self.T[i, :, k] = new_T_slice
        self.err += new_err_slice - old_err_slice
        return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    best = _schoolbook()
    best_rank = n * m * p
    
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]
    
    random.seed(12345)
    low_bound = max(n * m, m * p, p * n)
    rmax = best_rank
    
    for r in range(low_bound, rmax + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Increase restarts since local search is faster now
        for restart in range(100):
            if time.time() > deadline:
                break
                
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            density = random.choice([0.1, 0.2, 0.3, 0.5])
            
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < density:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < density:
                        V[idx, c] = random.choice(vs)
                for idx in range(p * n):
                    if random.random() < density:
                        W[idx, c] = random.choice(vs)
            
            searcher = _LocalSearcher(n, m, p, r, U, V, W)
            
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                if searcher.err < 1e-9:
                    break
                    
                mat_choice = random.choice([0, 1, 2])
                if mat_choice == 0:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    c = random.randint(0, r - 1)
                    new_val = random.choice(vs)
                    searcher.update_U(i, j, c, new_val)
                elif mat_choice == 1:
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    c = random.randint(0, r - 1)
                    new_val = random.choice(vs)
                    searcher.update_V(j, k, c, new_val)
                else:
                    k = random.randint(0, p - 1)
                    i = random.randint(0, n - 1)
                    c = random.randint(0, r - 1)
                    new_val = random.choice(vs)
                    searcher.update_W(k, i, c, new_val)
            
            if searcher.err < 1e-9:
                if _verify(searcher.U, searcher.V, searcher.W, n, m, p, r):
                    best = {"U": searcher.U.tolist(), "V": searcher.V.tolist(), "W": searcher.W.tolist()}
                    best_rank = r
                    found = True
                    break
        
        if found:
            return best
            
    return best
# EVOLVE-BLOCK-END
