    def _apply_x_rotation(state, n, beta):
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        # state is 1D array of size 2^n
        # We want to apply U_x on each qubit.
        # This is equivalent to:
        # new_state[i] = c * state[i] + s * state[i ^ mask_for_bit_k]
        # Wait, U_x is not just a bit flip. It mixes amplitudes.
        # U_x |0> = c|0> + s|1>
        # U_x |1> = s|0> + c|1>
        # So for each pair of states differing by bit k:
        # new[even] = c*old[even] + s*old[odd]
        # new[odd] = s*old[even] + c*old[odd]
        
        dim = 1 << n
        new_state = np.empty_like(state)
        for k in range(n):
            # Indices where bit k is 0 and 1
            # We can use reshape to (2, 2^(k), 2^(n-k-1), 2^(n-k-1))? No, standard trick:
            # Reshape to (2, 2^(k), 2^(n-k)) ?
            # Let's use the block structure.
            # For bit k, the state can be viewed as blocks of size 2^k.
            # Within each block of size 2^(k+1), the first half has bit k=0, second half bit k=1.
            block_size = 1 << (k + 1)
            half_size = 1 << k
            for start in range(0, dim, block_size):
                idx0 = slice(start, start + half_size)
                idx1 = slice(start + half_size, start + block_size)
                v0 = state[idx0]
                v1 = state[idx1]
                new_state[idx0] = c * v0 + s * v1
                new_state[idx1] = s * v0 + c * v1
        return new_state
    