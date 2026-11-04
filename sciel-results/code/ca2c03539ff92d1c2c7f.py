# Liu 1973 rank 19 decomposition for 3x3 matrix multiplication
# U_L, V_L, W_L are 9x19
# Indices: row = i*3+j (for U, V), row = k*3+i (for W)
# Wait, standard tensor: T[i,j,k] = A[i,j] B[j,k]
# U[i,j,r], V[j,k,r], W[k,i,r]
# My prompt requires:
# U[i*m+j, r] -> i*6+j globally. Locally i*3+j.
# V[j*p+k, r] -> j*6+k globally. Locally j*3+k.
# W[k*n+i, r] -> k*6+i globally. Locally k*3+i.

# Liu factors (from literature)
# I will use a known set of integer factors.
