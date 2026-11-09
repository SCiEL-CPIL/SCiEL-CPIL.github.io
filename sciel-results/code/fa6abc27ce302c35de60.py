# M2: (A10+A11)*B00
U[i_offset * m + (j_offset+1)][col] += 1  # labeled A10 but actually A[i_offset, j_offset+1]
U[(i_offset+1) * m + (j_offset+1)][col] += 1  # A[(i_offset+1), (j_offset+1)] = A11
V[j_offset * p + k_offset][col] += 1  # B[j_offset, k_offset] = B00
W[(k_offset+1) * n + (i_offset+1)][col] += 1  # C[i_offset+1, k_offset+1]
W[k_offset * n + i_offset][col] -= 1  # C[i_offset, k_offset]
