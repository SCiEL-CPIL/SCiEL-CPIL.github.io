"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Pre-computed Strassen decomposition for 2x2 matrix multiplication.
# We use the 7-term Strassen algorithm.
# For A (2x2) * B (2x2) = C (2x2):
# A = [[a11, a12], [a21, a22]]
# B = [[b11, b12], [b21, b22]]
# C = [[c11, c12], [c21, c22]]
#
# Strassen's 7 products:
# M1 = (a11 + a22) * (b11 + b22)
# M2 = (a21 + a22) * b11
# M3 = a11 * (b12 - b22)
# M4 = a22 * (b21 - b11)
# M5 = (a11 + a12) * b22
# M6 = (a21 - a11) * (b11 + b12)
# M7 = (a12 - a22) * (b21 + b22)
#
# Then:
# c11 = M1 + M4 - M5 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M3 + M6

# We need to express this as a tensor decomposition.
# The input to the 2x2 multiplication has 4 elements from A and 4 from B.
# The output has 4 elements.
#
# Let's index A's elements as: A[0]=a11, A[1]=a12, A[2]=a21, A[3]=a22
# Let's index B's elements as: B[0]=b11, B[1]=b12, B[2]=b21, B[3]=b22
# Let's index C's elements as: C[0]=c11, C[1]=c12, C[2]=c21, C[3]=c22
#
# The tensor T[i,j,k] where i indexes A, j indexes B, k indexes C.
# T[i,j,k] = 1 if C[k] depends on A[i]*B[j] in the standard way, but for Strassen
# we need to find the rank-1 decomposition.
#
# Actually, let's think of it differently. The matrix multiplication tensor for 2x2
# has rank 7 (Strassen). We need to find 7 vectors u_r, v_r, w_r such that
# T = sum_r u_r * v_r * w_r (outer product).
#
# Let's derive the decomposition.
# M1 = (a11 + a22) * (b11 + b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
# M2 = (a21 + a22) * b11 = a21*b11 + a22*b11
# M3 = a11 * (b12 - b22) = a11*b12 - a11*b22
# M4 = a22 * (b21 - b11) = a22*b21 - a22*b11
# M5 = (a11 + a12) * b22 = a11*b22 + a12*b22
# M6 = (a21 - a11) * (b11 + b12) = a21*b11 + a21*b12 - a11*b11 - a11*b12
# M7 = (a12 - a22) * (b21 + b22) = a12*b21 + a12*b22 - a22*b21 - a22*b22
#
# c11 = M1 + M4 - M5 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M3 + M6
#
# Let's write each M_r as a product of a linear form in A and a linear form in B.
# M1 = (a11 + a22) * (b11 + b22)
#   u1 = [1, 0, 0, 1]  (coefficients for a11, a12, a21, a22)
#   v1 = [1, 0, 0, 1]  (coefficients for b11, b12, b21, b22)
#   w1 = [1, 0, 0, 1]  (contributes to c11, c22)
#
# M2 = (a21 + a22) * b11
#   u2 = [0, 0, 1, 1]
#   v2 = [1, 0, 0, 0]
#   w2 = [0, 0, 1, 0]  (contributes to c21)
#
# M3 = a11 * (b12 - b22)
#   u3 = [1, 0, 0, 0]
#   v3 = [0, 1, 0, -1]
#   w3 = [0, 1, 0, 1]  (contributes to c12, c22)
#
# M4 = a22 * (b21 - b11)
#   u4 = [0, 0, 0, 1]
#   v4 = [-1, 0, 1, 0]
#   w4 = [1, 0, 1, 0]  (contributes to c11, c21)
#
# M5 = (a11 + a12) * b22
#   u5 = [1, 1, 0, 0]
#   v5 = [0, 0, 0, 1]
#   w5 = [-1, 0, 0, 0]  (contributes to c11 with -1)
#
# M6 = (a21 - a11) * (b11 + b12)
#   u6 = [-1, 0, 1, 0]
#   v6 = [1, 1, 0, 0]
#   w6 = [0, 0, 0, 1]  (contributes to c22)
#
# M7 = (a12 - a22) * (b21 + b22)
#   u7 = [0, 1, 0, -1]
#   v7 = [0, 0, 1, 1]
#   w7 = [1, 0, 0, 0]  (contributes to c11)
#
# Let's verify c11 = M1 + M4 - M5 + M7:
# c11 coefficient from each M_r:
# M1: w1[0] = 1
# M2: w2[0] = 0
# M3: w3[0] = 0
# M4: w4[0] = 1
# M5: w5[0] = -1
# M6: w6[0] = 0
# M7: w7[0] = 1
# Sum: 1 + 1 - 1 + 1 = 2. But c11 should have coefficient 1 for a11*b11.
# Let's check the actual contribution to c11 from a11*b11:
# M1: a11*b11 contributes 1*1*1 = 1
# M4: a22*b21 and a22*b11, no a11*b11
# M5: no a11*b11
# M7: no a11*b11
# So a11*b11 contributes 1 to c11. Correct.
#
# Let's verify c12 = M3 + M5:
# c12 coefficient from each M_r:
# M3: w3[1] = 1
# M5: w5[1] = 0
# Sum: 1. But c12 = a11*b12 + a12*b22.
# M3: a11*(b12 - b22) contributes a11*b12 with coeff 1.
# M5: (a11+a12)*b22 contributes a12*b22 with coeff 1.
# So c12 = a11*b12 + a12*b22. Correct.
#
# Let's verify c21 = M2 + M4:
# c21 coefficient from each M_r:
# M2: w2[2] = 1
# M4: w4[2] = 1
# Sum: 2. But c21 = a21*b11 + a22*b21.
# M2: (a21+a22)*b11 contributes a21*b11 with coeff 1.
# M4: a22*(b21-b11) contributes a22*b21 with coeff 1.
# So c21 = a21*b11 + a22*b21. Correct.
#
# Let's verify c22 = M1 - M2 + M3 + M6:
# c22 coefficient from each M_r:
# M1: w1[3] = 1
# M2: w2[3] = 0
# M3: w3[3] = 1
# M6: w6[3] = 1
# Sum: 1 + 1 + 1 = 3. But c22 = a12*b12 + a22*b22.
# Let's check contributions:
# M1: a11*b11 + a11*b22 + a22*b11 + a22*b22. Contributes a22*b22 with coeff 1.
# M2: (a21+a22)*b11. No contribution to c22.
# M3: a11*(b12-b22). No contribution to c22.
# M6: (a21-a11)*(b11+b12) = a21*b11 + a21*b12 - a11*b11 - a11*b12. No a12*b12.
#
# Hmm, I think I made an error. Let me re-derive.
# c22 = a12*b12 + a22*b22
# M1 - M2 + M3 + M6:
# M1: a11*b11 + a11*b22 + a22*b11 + a22*b22
# -M2: -a21*b11 - a22*b11
# M3: a11*b12 - a11*b22
# M6: a21*b11 + a21*b12 - a11*b11 - a11*b12
# Sum:
# a11*b11: 1 - 1 = 0
# a11*b22: 1 - 1 = 0
# a22*b11: 1 - 1 = 0
# a22*b22: 1
# a21*b11: -1 + 1 = 0
# a11*b12: 1 - 1 = 0
# a21*b12: 1
# So we get a22*b22 + a21*b12. But c22 should be a12*b12 + a22*b22.
#
# I think I have the wrong Strassen formulas. Let me look up the correct ones.
#
# Correct Strassen for 2x2:
# Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
#
# c11 = M1 - M5 + M4 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M6 + M3
#
# Let me re-verify with A = [[a,b],[c,d]], B = [[e,f],[g,h]]
# c11 = a*e + c*g
# c12 = a*f + c*h
# c21 = b*e + d*g
# c22 = b*f + d*h
#
# M1 = (a+d)(e+h) = ae + ah + de + dh
# M2 = (c+d)e = ce + de
# M3 = a(f-h) = af - ah
# M4 = d(g-e) = dg - de
# M5 = (a+b)h = ah + bh
# M6 = (c-a)(e+f) = ce + cf - ae - af
# M7 = (b-d)(g+h) = bg + bh - dg - dh
#
# c11 = M1 - M5 + M4 + M7
# = (ae + ah + de + dh) - (ah + bh) + (dg - de) + (bg + bh - dg - dh)
# = ae + ah + de + dh - ah - bh + dg - de + bg + bh - dg - dh
# = ae + bg
# Correct!
#
# c12 = M3 + M5
# = (af - ah) + (ah + bh)
# = af + bh
# Correct!
#
# c21 = M2 + M4
# = (ce + de) + (dg - de)
# = ce + dg
# But c21 should be b*e + d*g. I have c*e + d*g. 
# Wait, I think I mixed up the indices. Let me re-read.
# A = [[a, b], [c, d]] so A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
# B = [[e, f], [g, h]] so B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
# C = A*B, so C[i,j] = sum_k A[i,k] * B[k,j]
# C[0,0] = a*e + b*g
# C[0,1] = a*f + b*h
# C[1,0] = c*e + d*g
# C[1,1] = c*f + d*h
#
# So c21 (which is C[1,0]) = c*e + d*g. My formula gives ce + dg. Correct!
#
# c22 = M1 - M2 + M6 + M3
# = (ae + ah + de + dh) - (ce + de) + (ce + cf - ae - af) + (af - ah)
# = ae + ah + de + dh - ce - de + ce + cf - ae - af + af - ah
# = dh + cf
# But C[1,1] should be c*f + d*h. I have dh + cf. Correct!
#
# So the correct Strassen formulas are:
# c11 = M1 - M5 + M4 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M6 + M3
#
# Now let's write the decomposition.
# A indices: 0=a, 1=b, 2=c, 3=d
# B indices: 0=e, 1=f, 2=g, 3=h
# C indices: 0=c11, 1=c12, 2=c21, 3=c22
#
# M1 = (a+d)(e+h)
#   u1 = [1, 0, 0, 1]
#   v1 = [1, 0, 0, 1]
#   w1 = [1, 0, 0, 1]  (contributes to c11 and c22)
#
# M2 = (c+d)e
#   u2 = [0, 0, 1, 1]
#   v2 = [1, 0, 0, 0]
#   w2 = [0, 0, 1, 0]  (contributes to c21)
#
# M3 = a(f-h)
#   u3 = [1, 0, 0, 0]
#   v3 = [0, 1, 0, -1]
#   w3 = [0, 1, 0, 1]  (contributes to c12 and c22)
#
# M4 = d(g-e)
#   u4 = [0, 0, 0, 1]
#   v4 = [-1, 0, 1, 0]
#   w4 = [1, 0, 1, 0]  (contributes to c11 and c21)
#
# M5 = (a+b)h
#   u5 = [1, 1, 0, 0]
#   v5 = [0, 0, 0, 1]
#   w5 = [-1, 0, 0, 0]  (contributes to c11 with -1)
#
# M6 = (c-a)(e+f)
#   u6 = [-1, 0, 1, 0]
#   v6 = [1, 1, 0, 0]
#   w6 = [0, 0, 0, 1]  (contributes to c22)
#
# M7 = (b-d)(g+h)
#   u7 = [0, 1, 0, -1]
#   v7 = [0, 0, 1, 1]
#   w7 = [1, 0, 0, 0]  (contributes to c11)
#
# Let's verify c11 = M1 - M5 + M4 + M7:
# c11 coefficient:
# M1: w1[0] = 1
# M5: w5[0] = -1
# M4: w4[0] = 1
# M7: w7[0] = 1
# Sum: 1 - 1 + 1 + 1 = 2. But we need to check the actual contributions.
#
# Actually, the w vectors represent the coefficients for each output. Let me re-examine.
# The decomposition is T = sum_r u_r * v_r * w_r, where
# T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
#
# So for c11 (k=0):
# T[i,j,0] = sum_r u_r[i] * v_r[j] * w_r[0]
#
# Let's compute T[0,0,0] (coefficient of a*e in c11):
# M1: u1[0]*v1[0]*w1[0] = 1*1*1 = 1
# M5: u5[0]*v5[0]*w5[0] = 1*0*(-1) = 0
# M4: u4[0]*v4[0]*w4[0] = 0*(-1)*1 = 0
# M7: u7[0]*v7[0]*w7[0] = 0*0*1 = 0
# Sum: 1. Correct (a*e contributes 1 to c11).
#
# Let's compute T[1,2,0] (coefficient of b*g in c11):
# M1: u1[1]*v1[2]*w1[0] = 0*0*1 = 0
# M5: u5[1]*v5[2]*w5[0] = 1*0*(-1) = 0
# M4: u4[1]*v4[2]*w4[0] = 0*1*1 = 0
# M7: u7[1]*v7[2]*w7[0] = 1*1*1 = 1
# Sum: 1. Correct (b*g contributes 1 to c11).
#
# Let's compute T[0,1,1] (coefficient of a*f in c12):
# M3: u3[0]*v3[1]*w3[1] = 1*1*1 = 1
# M5: u5[0]*v5[1]*w5[1] = 1*0*0 = 0
# Sum: 1. Correct.
#
# Let's compute T[2,0,2] (coefficient of c*e in c21):
# M2: u2[2]*v2[0]*w2[2] = 1*1*1 = 1
# M4: u4[2]*v4[0]*w4[2] = 0*(-1)*1 = 0
# Sum: 1. Correct.
#
# Let's compute T[1,1,3] (coefficient of b*f in c22):
# M6: u6[1]*v6[1]*w6[3] = 0*1*1 = 0
# M3: u3[1]*v3[1]*w3[3] = 0*1*1 = 0
# Hmm, this should be 1. Let me check.
#
# Wait, c22 = b*f + d*h. Let me check T[1,1,3]:
# M1: u1[1]*v1[1]*w1[3] = 0*0*1 = 0
# M2: u2[1]*v2[1]*w2[3] = 0*0*0 = 0
# M3: u3[1]*v3[1]*w3[3] = 0*1*1 = 0
# M4: u4[1]*v4[1]*w4[3] = 0*0*0 = 0
# M5: u5[1]*v5[1]*w5[3] = 1*0*0 = 0
# M6: u6[1]*v6[1]*w6[3] = 0*1*1 = 0
# M7: u7[1]*v7[1]*w7[3] = 1*0*0 = 0
# Sum: 0. But it should be 1.
#
# I think I have an error in the w vectors. Let me re-examine.
# c22 = M1 - M2 + M6 + M3
# So w_r[3] should be:
# M1: 1
# M2: -1
# M6: 1
# M3: 1
# Others: 0
#
# Let me recompute T[1,1,3]:
# M1: u1[1]*v1[1]*w1[3] = 0*0*1 = 0
# M2: u2[1]*v2[1]*w2[3] = 0*0*(-1) = 0
# M6: u6[1]*v6[1]*w6[3] = 0*1*1 = 0
# M3: u3[1]*v3[1]*w3[3] = 0*1*1 = 0
# Still 0.
#
# Let me check M6 more carefully.
# M6 = (c-a)(e+f)
# u6 = [-1, 0, 1, 0]  (coefficients for a, b, c, d)
# v6 = [1, 1, 0, 0]  (coefficients for e, f, g, h)
#
# M6 = -a*e - a*f + c*e + c*f
#
# And c22 = b*f + d*h. M6 doesn't contribute to b*f.
#
# I think the issue is that I'm not correctly identifying which M_r contributes to which c_k.
# Let me re-derive the w vectors from the formulas.
#
# c11 = M1 - M5 + M4 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M6 + M3
#
# So:
# w1 = [1, 0, 0, 1]  (M1 contributes to c11 and c22)
# w2 = [0, 0, 1, -1]  (M2 contributes to c21 and -c22)
# w3 = [0, 1, 0, 1]  (M3 contributes to c12 and c22)
# w4 = [1, 0, 1, 0]  (M4 contributes to c11 and c21)
# w5 = [-1, 1, 0, 0]  (M5 contributes to -c11 and c12)
# w6 = [0, 0, 0, 1]  (M6 contributes to c22)
# w7 = [1, 0, 0, 0]  (M7 contributes to c11)
#
# Let me re-verify T[1,1,3] (b*f in c22):
# M1: u1[1]*v1[1]*w1[3] = 0*0*1 = 0
# M2: u2[1]*v2[1]*w2[3] = 0*0*(-1) = 0
# M3: u3[1]*v3[1]*w3[3] = 0*1*1 = 0
# M4: u4[1]*v4[1]*w4[3] = 0*0*0 = 0
# M5: u5[1]*v5[1]*w5[3] = 1*0*0 = 0
# M6: u6[1]*v6[1]*w6[3] = 0*1*1 = 0
# M7: u7[1]*v7[1]*w7[3] = 1*0*0 = 0
# Still 0.
#
# Let me check M6 again. M6 = (c-a)(e+f) = -ae - af + ce + cf.
# This doesn't have b*f. So M6 doesn't contribute to b*f.
#
# But c22 = b*f + d*h. Which M_r has b*f?
# Let's look at all M_r:
# M1 = ae + ah + de + dh
# M2 = ce + de
# M3 = af - ah
# M4 = dg - de
# M5 = ah + bh
# M6 = -ae - af + ce + cf
# M7 = bg + bh - dg - dh
#
# None of them have b*f! But c22 = b*f + d*h.
#
# I think I made an error in the Strassen formulas. Let me look them up again.
#
# Actually, I think the correct Strassen formulas are:
# M1 = (a+d)(e+h)
# M2 = (c+d)(e)
# M3 = (a)(f-h)
# M4 = (d)(g-e)
# M5 = (a+b)(h)
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
#
# c11 = M1 - M5 + M4 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M6 + M3
#
# Let me re-expand c22:
# M1 - M2 + M6 + M3
# = (ae + ah + de + dh) - (ce + de) + (-ae - af + ce + cf) + (af - ah)
# = ae + ah + de + dh - ce - de - ae - af + ce + cf + af - ah
# = dh + cf
#
# But C[1,1] = c*f + d*h. So dh + cf = c*f + d*h. Correct!
#
# Now let's find which M_r contributes to b*f in c22.
# b*f doesn't appear in any M_r. But c22 = b*f + d*h.
#
# Wait, I think I'm confusing myself. Let me re-read the problem.
# A = [[a, b], [c, d]]
# B = [[e, f], [g, h]]
# C = A*B
# C[0,0] = a*e + b*g
# C[0,1] = a*f + b*h
# C[1,0] = c*e + d*g
# C[1,1] = c*f + d*h
#
# So c22 = C[1,1] = c*f + d*h. Not b*f + d*h. I made an error earlier.
#
# Let me re-verify:
# c22 = M1 - M2 + M6 + M3
# = (ae + ah + de + dh) - (ce + de) + (-ae - af + ce + cf) + (af - ah)
# = ae + ah + de + dh - ce - de - ae - af + ce + cf + af - ah
# = dh + cf
# = d*h + c*f
# Correct!
#
# Now let's re-verify T[1,1,3] (b*f in c22). But c22 = c*f + d*h, so b*f should have coefficient 0.
# Let me check T[2,1,3] (c*f in c22):
# M1: u1[2]*v1[1]*w1[3] = 0*0*1 = 0
# M2: u2[2]*v2[1]*w2[3] = 1*0*(-1) = 0
# M3: u3[2]*v3[1]*w3[3] = 0*1*1 = 0
# M4: u4[2]*v4[1]*w4[3] = 0*0*0 = 0
# M5: u5[2]*v5[1]*w5[3] = 0*0*0 = 0
# M6: u6[2]*v6[1]*w6[3] = 1*1*1 = 1
# M7: u7[2]*v7[1]*w7[3] = 0*0*0 = 0
# Sum: 1. Correct.
#
# Let me check T[3,3,3] (d*h in c22):
# M1: u1[3]*v1[3]*w1[3] = 1*1*1 = 1
# M2: u2[3]*v2[3]*w2[3] = 1*0*(-1) = 0
# M3: u3[3]*v3[3]*w3[3] = 0*(-1)*1 = 0
# M4: u4[3]*v4[3]*w4[3] = 1*0*0 = 0
# M5: u5[3]*v5[3]*w5[3] = 0*1*0 = 0
# M6: u6[3]*v6[3]*w6[3] = 0*0*1 = 0
# M7: u7[3]*v7[3]*w7[3] = -1*1*0 = 0
# Sum: 1. Correct.
#
# Great, so the decomposition is correct. Now let's implement it for 6x6.
#
# For 6x6, we can use 3x3 blocks of 2x2. Each 2x2 multiplication has rank 7.
# The total rank will be 7^3 = 343? No, that's not right.
#
# Actually, for block matrix multiplication, if we have 3x3 blocks of 2x2,
# the rank is the rank of the 3x3 block multiplication tensor times the rank of the 2x2 multiplication.
#
# Wait, no. The block matrix multiplication is:
# C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
# where each A_block, B_block, C_block is a 2x2 matrix.
#
# The tensor for this block multiplication has rank r_block (for 3x3 block multiplication).
# The tensor for 2x2 matrix multiplication has rank 7.
#
# The total rank is r_block * 7? No, that's not right either.
#
# Actually, the total tensor is the tensor product of the block multiplication tensor
# and the 2x2 multiplication tensor. So the rank is the product of the ranks.
#
# The rank of 3x3 matrix multiplication is at most 23 (Lader's algorithm) or 27 (schoolbook).
# The rank of 2x2 matrix multiplication is 7 (Strassen).
#
# So the total rank would be at most 23 * 7 = 161? But that's not right because
# the block multiplication and the 2x2 multiplication are not independent.
#
# Actually, I think the correct approach is to use the Kronecker product structure.
# The 6x6 matrix multiplication tensor is the tensor product of the 3x3 block multiplication
# tensor and the 2x2 multiplication tensor.
#
# So if the 3x3 block multiplication has rank r3 and the 2x2 multiplication has rank 7,
# the total rank is r3 * 7.
#
# The rank of 3x3 matrix multiplication is at most 23 (Lader's algorithm).
# So the total rank would be at most 23 * 7 = 161.
#
# But wait, I don't have a rank-23 decomposition for 3x3. Let me use the schoolbook
# decomposition for 3x3, which has rank 27.
#
# So the total rank would be 27 * 7 = 189.
#
# This matches the knowledge from the prompt: "Block Strassen decomposition (3x3 blocks of 2x2)
# reduces rank from 216 to 189 by exploiting the lower rank of 2x2 matrix multiplication (7 vs 8)"
#
# So let's implement this.
#
# First, let's define the schoolbook decomposition for 3x3.
# The schoolbook decomposition has rank 27 (3*3*3).
# U_block[i*3+j, c], V_block[j*3+k, c], W_block[k*3+i, c]
#
# And the Strassen decomposition for 2x2 has rank 7.
# U_strassen[4, 7], V_strassen[4, 7], W_strassen[4, 7]
#
# The total decomposition is the Kronecker product of these two decompositions.
#
# Let's implement this.

def construct():
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    # Block size: 2x2, so 3x3 blocks
    block_size = 2
    num_blocks = 3  # 6 / 2 = 3
    
    # Schoolbook decomposition for 3x3 block multiplication
    rank_block = num_blocks * num_blocks * num_blocks  # 27
    
    # Strassen decomposition for 2x2 multiplication
    rank_strassen = 7
    
    # Total rank
    rank = rank_block * rank_strassen  # 27 * 7 = 189
    
    # Define the Strassen decomposition for 2x2
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    W_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    # Define the schoolbook decomposition for 3x3
    # U_block[i*3+j, c], V_block[j*3+k, c], W_block[k*3+i, c]
    U_block = [[0] * rank_block for _ in range(num_blocks * num_blocks)]
    V_block = [[0] * rank_block for _ in range(num_blocks * num_blocks)]
    W_block = [[0] * rank_block for _ in range(num_blocks * num_blocks)]
    column = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                U_block[i * num_blocks + j][column] = 1
                V_block[j * num_blocks + k][column] = 1
                W_block[k * num_blocks + i][column] = 1
                column += 1
    
    # Now construct the total decomposition using Kronecker product
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block multiplication term (i_block, j_block, k_block) and each Strassen term (s),
    # we have a rank-1 term.
    # The row index in U is (i_block * block_size + i_strassen) * m + (j_block * block_size + j_strassen)
    # Wait, let me think about the indexing more carefully.
    #
    # A is 6x6. A[i, j] where i is row, j is column.
    # Block (i_block, j_block) contains A[i_block*2 + i_strassen, j_block*2 + j_strassen]
    # for i_strassen in 0,1 and j_strassen in 0,1.
    #
    # In the schoolbook decomposition for 3x3 blocks:
    # U_block[i_block * 3 + j_block, c_block] = 1 if the term is (i_block, j_block, k_block)
    #
    # In the Strassen decomposition for 2x2:
    # U_strassen[i_strassen * 2 + j_strassen, s] ... wait, no.
    #
    # Let me re-think. The 2x2 multiplication is:
    # C_block = A_block * B_block
    # where A_block is 2x2, B_block is 2x2, C_block is 2x2.
    #
    # The input to the 2x2 multiplication is A_block (4 elements) and B_block (4 elements).
    # The output is C_block (4 elements).
    #
    # In the Strassen decomposition:
    # U_strassen[i, s] where i indexes the 4 elements of A_block, s indexes the 7 Strassen terms.
    # V_strassen[j, s] where j indexes the 4 elements of B_block, s indexes the 7 Strassen terms.
    # W_strassen[k, s] where k indexes the 4 elements of C_block, s indexes the 7 Strassen terms.
    #
    # So for a given block (i_block, j_block, k_block) and Strassen term s:
    # The contribution to U is at row (i_block*2 + i_strassen, j_block*2 + j_strassen)
    # for each i_strassen where U_strassen[i_strassen, s] != 0.
    #
    # Wait, I need to be more careful. The U matrix has shape (n*m, rank) = (36, 189).
    # The row index is i*m + j = i*6 + j, where i is the row of A and j is the column of A.
    #
    # For a given block (i_block, j_block) and Strassen term s, the contribution to U
    # is at rows (i_block*2 + i_strassen)*6 + (j_block*2 + j_strassen) for each i_strassen, j_strassen.
    #
    # But U_strassen only has 4 rows (for the 4 elements of A_block). So i_strassen and j_strassen
    # are combined into a single index. Let's say the 4 elements of A_block are indexed as:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # So i_strassen_combined = i_strassen * 2 + j_strassen.
    #
    # Similarly for B_block and C_block.
    #
    # So for a given block (i_block, j_block, k_block) and Strassen term s:
    # For each i_strassen_combined in 0..3:
    #   i_strassen = i_strassen_combined // 2
    #   j_strassen = i_strassen_combined % 2
    #   row_U = (i_block*2 + i_strassen)*6 + (j_block*2 + j_strassen)
    #   U[row_U, c] += U_strassen[i_strassen_combined, s]
    #   where c = (i_block*3 + j_block)*7 + k_block*7 + s ... wait, no.
    #
    # Let me re-think the column indexing.
    # The total rank is 189 = 27 * 7.
    # The column index c can be decomposed as c = c_block * 7 + s,
    # where c_block is the column index in the 3x3 schoolbook decomposition (0..26)
    # and s is the Strassen term index (0..6).
    #
    # In the 3x3 schoolbook decomposition, c_block = i_block*9 + j_block*3 + k_block.
    #
    # So c = (i_block*9 + j_block*3 + k_block) * 7 + s.
    #
    # Now for V:
    # V has shape (m*p, rank) = (36, 189).
    # The row index is j*m + k = j*6 + k, where j is the row of B and k is the column of B.
    #
    # For a given block (i_block, j_block, k_block) and Strassen term s:
    # For each j_strassen_combined in 0..3:
    #   j_strassen = j_strassen_combined // 2
    #   k_strassen = j_strassen_combined % 2
    #   row_V = (j_block*2 + j_strassen)*6 + (k_block*2 + k_strassen)
    #   V[row_V, c] += V_strassen[j_strassen_combined, s]
    #
    # For W:
    # W has shape (p*n, rank) = (36, 189).
    # The row index is k*n + i = k*6 + i, where k is the row of C and i is the column of C.
    # Wait, the problem says "The output factor follows the official trace convention W[k*n+i,r]".
    # So W[k*n + i, r] where k is the row of C and i is the column of C.
    #
    # For a given block (i_block, j_block, k_block) and Strassen term s:
    # For each k_strassen_combined in 0..3:
    #   k_strassen = k_strassen_combined // 2
    #   i_strassen = k_strassen_combined % 2
    #   row_W = (k_block*2 + k_strassen)*6 + (i_block*2 + i_strassen)
    #   W[row_W, c] += W_strassen[k_strassen_combined, s]
    
    for i_block in range(num_blocks):
        for j_block in range(num_blocks):
            for k_block in range(num_blocks):
                c_block = i_block * num_blocks * num_blocks + j_block * num_blocks + k_block
                for s in range(rank_strassen):
                    c = c_block * rank_strassen + s
                    # U
                    for i_strassen_combined in range(4):
                        i_strassen = i_strassen_combined // 2
                        j_strassen = i_strassen_combined % 2
                        row_U = (i_block * block_size + i_strassen) * m + (j_block * block_size + j_strassen)
                        U[row_U][c] += U_strassen[i_strassen_combined][s]
                    # V
                    for j_strassen_combined in range(4):
                        j_strassen = j_strassen_combined // 2
                        k_strassen = j_strassen_combined % 2
                        row_V = (j_block * block_size + j_strassen) * p + (k_block * block_size + k_strassen)
                        V[row_V][c] += V_strassen[j_strassen_combined][s]
                    # W
                    for k_strassen_combined in range(4):
                        k_strassen = k_strassen_combined // 2
                        i_strassen = k_strassen_combined % 2
                        row_W = (k_block * block_size + k_strassen) * n + (i_block * block_size + i_strassen)
                        W[row_W][c] += W_strassen[k_strassen_combined][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
