def add_block(U, V, W, I, J, K, block_rank, block_U, block_V, block_W):
    # I, J, K are lists of global indices
    # block_U is local_U (len(I) x block_rank)
    # block_V is local_V (len(K) x block_rank) -- Wait, V is K x J?
    # In seed: V[j, k]. j is inner, k is col.
    # So V is Inner x Col.
    # block_V should be (len(K) x len(J))?
    # My previous analysis: A is I x K. B is K x J.
    # So block_U is (len(I) x block_rank).
    # block_V is (len(K) x block_rank).
    # block_W is (len(J) x block_rank).
    
    for r in range(block_rank):
        for i_idx, i_val in enumerate(I):
            for k_idx, k_val in enumerate(K):
                U[i_val * 5 + k_val][r] += block_U[i_idx][r]
        for k_idx, k_val in enumerate(K):
            for j_idx, j_val in enumerate(J):
                V[k_val * 5 + j_val][r] += block_V[k_idx][r]
        for j_idx, j_val in enumerate(J):
            for i_idx, i_val in enumerate(I):
                W[j_val * 5 + i_val][r] += block_W[j_idx][r]
