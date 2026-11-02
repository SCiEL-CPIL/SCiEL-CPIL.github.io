# Q = A[0,2] * (B[2,0] + ... + B[2,8]) -- no wait
# We need: for each k, C[0,k] += A[0,2]*B[2,k] and C[1,k] += A[1,2]*B[2,k]
# 
# With rank 2:
# r0: U[2, r0]=1, V[18+k, r0]=1 for all k, W[k*2+0, r0]=1, W[k*2+1, r0]=0
# r1: U[5, r1]=1, V[18+k, r1]=1 for all k, W[k*2+0, r1]=0, W[k*2+1, r1]=1
