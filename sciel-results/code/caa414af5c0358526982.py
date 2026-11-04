   def _kronecker_decomposition(n, m, p):
       U2, V2, W2 = _strassen_2x2()
       U2 = np.array(U2, dtype=float)
       V2 = np.array(V2, dtype=float)
       W2 = np.array(W2, dtype=float)
       
       # Reshape to (2, 2, 7)
       # Index a = 2*i + k
       U2_3d = U2.reshape(2, 2, 7) # Wait, reshape order?
       # U2 is (4, 7).
       # We want U2_3d[i, k, r] = U2[2*i+k, r].
       # Standard C-order reshape of (4,7) to (2,2,7) gives:
       # Element [0,0,0] is U2[0,0]
       # Element [0,1,0] is U2[1,0]
       # Element [1,0,0] is U2[2,0]
       # Element [1,1,0] is U2[3,0]
       # So U2_3d[i, k, r] = U2[2*i+k, r]. This matches.
       
       U2_3d = U2.reshape(2, 2, 7)
       V2_3d = V2.reshape(2, 2, 7)
       W2_3d = W2.reshape(2, 2, 7)
       
       # Compute Kronecker product
       # U4[i1, i2, k1, k2, r1, r2] = U2_3d[i1, k1, r1] * U2_3d[i2, k2, r2]
       # We want U4_flat[I, K, R] where I=2*i1+i2, K=2*k1+k2, R=7*r1+r2.
       
       # Use einsum to compute the 6D tensor
       U4_6d = np.einsum('ikr,jls->ijklrs', U2_3d, U2_3d)
       V4_6d = np.einsum('ikr,jls->ijklrs', V2_3d, V2_3d)
       W4_6d = np.einsum('ikr,jls->ijklrs', W2_3d, W2_3d)
       
       # Reshape to (4, 4, 49)
       # Axis order: i1, i2, k1, k2, r1, r2
       # We want I (i1,i2), K (k1,k2), R (r1,r2).
       # So transpose to (i1, i2, k1, k2, r1, r2) is already that order.
       # Reshape (2, 2, 2, 2, 7, 7) to (4, 4, 49).
       U4 = U4_6d.reshape(4, 4, 49)
       V4 = V4_6d.reshape(4, 4, 49)
       W4 = W4_6d.reshape(4, 4, 49)
       
       # Convert to lists of lists
       return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist(), "rank": 49}
   