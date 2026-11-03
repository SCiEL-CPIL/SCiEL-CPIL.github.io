<<<<<<< SEARCH
    # --- 3. Helper to embed and combine decompositions ---
    def combine_decomps(decomps):
        """
        Combine a list of decompositions.
        Each decomposition is (n, m, p, U, V, W, offset_i, offset_j, offset_k)
        Returns global U, V, W.
        """
        if not decomps:
            return [], [], []
        
        global_n = max(n for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        global_m = max(m for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        global_p = max(p for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        
        # Determine total rank
        total_rank = sum(len(U) for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        
        G_U = [[0] * total_rank for _ in range(global_n * global_m)]
        G_V = [[0] * total_rank for _ in range(global_m * global_p)]
        G_W = [[0] * total_rank for _ in range(global_p * global_n)]
        
        rank_offset = 0
        for (n, m, p, U, V, W, oi, oj, ok) in decomps:
            for c in range(len(U)):
                r_c = rank_offset + c
                # U mapping: local row (i*m + j) -> global row ((oi+i)*global_m + (oj+j))
                for i in range(n):
                    for j in range(m):
                        local_u_idx = i * m + j
                        global_u_idx = (oi + i) * global_m + (oj + j)
                        G_U[global_u_idx][r_c] = U[local_u_idx][c]
                        
                # V mapping: local row (j*p + k) -> global row ((oj+j)*global_p + (ok+k))
                for j in range(m):
                    for k in range(p):
                        local_v_idx = j * p + k
                        global_v_idx = (oj + j) * global_p + (ok + k)
                        G_V[global_v_idx][r_c] = V[local_v_idx][c]
                        
                # W mapping: local row (k*n + i) -> global row ((ok+k)*global_n + (oi+i))
                for k in range(p):
                    for i in range(n):
                        local_w_idx = k * n + i
                        global_w_idx = (ok + k) * global_n + (oi + i)
                        G_W[global_w_idx][r_c] = W[local_w_idx][c]
                        
            rank_offset += len(U)
            
        return G_U, G_V, G_W
=======
    # --- 3. Helper to embed and combine decompositions ---
    def combine_decomps(decomps, global_n, global_m, global_p):
        """
        Combine a list of decompositions.
        Each decomposition is (n, m, p, U, V, W, offset_i, offset_j, offset_k)
        Returns global U, V, W.
        """
        if not decomps:
            return [], [], []
        
        # Determine total rank
        total_rank = sum(len(U) for (n, m, p, U, V, W, oi, oj, ok) in decomps)
        
        G_U = [[0] * total_rank for _ in range(global_n * global_m)]
        G_V = [[0] * total_rank for _ in range(global_m * global_p)]
        G_W = [[0] * total_rank for _ in range(global_p * global_n)]
        
        rank_offset = 0
        for (n, m, p, U, V, W, oi, oj, ok) in decomps:
            for c in range(len(U)):
                r_c = rank_offset + c
                # U mapping: local row (i*m + j) -> global row ((oi+i)*global_m + (oj+j))
                for i in range(n):
                    for j in range(m):
                        local_u_idx = i * m + j
                        global_u_idx = (oi + i) * global_m + (oj + j)
                        G_U[global_u_idx][r_c] = U[local_u_idx][c]
                        
                # V mapping: local row (j*p + k) -> global row ((oj+j)*global_p + (ok+k))
                for j in range(m):
                    for k in range(p):
                        local_v_idx = j * p + k
                        global_v_idx = (oj + j) * global_p + (ok + k)
                        G_V[global_v_idx][r_c] = V[local_v_idx][c]
                        
                # W mapping: local row (k*n + i) -> global row ((ok+k)*global_n + (oi+i))
                for k in range(p):
                    for i in range(n):
                        local_w_idx = k * n + i
                        global_w_idx = (ok + k) * global_n + (oi + i)
                        G_W[global_w_idx][r_c] = W[local_w_idx][c]
                        
            rank_offset += len(U)
            
        return G_U, G_V, G_W
>>>>>>> REPLACE

