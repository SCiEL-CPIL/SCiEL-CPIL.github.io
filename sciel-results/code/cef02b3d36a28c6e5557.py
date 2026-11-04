# Part 2: 2x1x8 - rank 1 (outer product)
base_r = 28
r = base_r
# U[i*3+2, r] = 1 for i=0,1
U[2][r] = 1
U[5][r] = 1
# V[2*8+k, r] = 1 for k=0..7
for k in range(8):
    V[16+k][r] = 1
# W[k*2+i, r] = 1 for k=0..7, i=0,1
for k in range(8):
    for i in range(2):
        W[k*2+i][r] = 1
