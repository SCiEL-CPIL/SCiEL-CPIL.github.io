# M7 = (A[a_r0,b_c0]+A[a_r0,b_c1])*B[b_c1,b_k0+1]
U[a_r0*3+b_c0][base+6] += 1; V[b_c1*8+b_k0+1][base+6] += 1
U[a_r0*3+b_c1][base+6] += 1; V[b_c1*8+b_k0+1][base+6] += 0
W[b_k0*2+a_r0][base+6] += 1; W[(b_k0+1)*2+a_r1][base+6] += 1
