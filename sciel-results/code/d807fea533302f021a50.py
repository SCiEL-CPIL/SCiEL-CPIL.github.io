for (br, bc, coeff) in c_targets[t]:
    W[w_idx(br, bc, 0, 0), col] += coeff
    W[w_idx(br, bc, 1, 1), col] += coeff
