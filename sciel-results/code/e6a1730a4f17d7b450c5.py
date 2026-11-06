def _compute_energy_fast(state, n, coupling, field, zz_signs, x_flip_indices):
    prob = np.abs(state) ** 2
    e_zy = 0.0
    for i in range(n-1):
        e_zy += -coupling * np.dot(prob, zz_signs[i])
        
    e_x = 0.0
    for i in range(n):
        # <X_i> = sum conj(state[s]) state[s^bit_i]
        # We can use the precomputed indices
        flipped = state[x_flip_indices[i]]
        e_x += -field * np.real(np.dot(np.conj(state), flipped))
        
    return e_zy + e_x
