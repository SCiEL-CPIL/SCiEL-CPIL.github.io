       def construct():
           # ...
           # Rank 23 3x3x3 decomposition
           U33 = np.array([...])
           V33 = np.array([...])
           W33 = np.array([...])
           
           # Rank 9 3x3x1 decomposition
           U31 = np.zeros((9, 9))
           V31 = np.zeros((9, 9)) # Wait, V for 3x1 is 3x1 inputs? No.
           # For 3x3 * 3x1:
           # U is 9x9 (i,j)
           # V is 3x9 (j, k=0) -> Row index j
           # W is 3x9 (k=0, i) -> Row index i
           
           # Combine for 3x3 * 3x4
           # U: 9 x 32
           # V: 12 x 32
           # W: 12 x 32
           
           U = np.zeros((9, 32))
           V = np.zeros((12, 32))
           W = np.zeros((12, 32))
           
           # Part 1: 3x3 * 3x3 (cols 0,1,2 of B)
           # U[:, 0:23] = U33
           # V[0:9, 0:23] = V33 (Rows 0..8 correspond to j=0..2, k=0..2)
           # W[0:9, 0:23] = W33 (Rows 0..8 correspond to k=0..2, i=0..2)
           
           # Part 2: 3x3 * 3x1 (col 3 of B)
           # U[:, 23:32] = U31
           # V[9:12, 23:32] = V31 (Rows 9..11 correspond to j=0..2, k=3)
           # W[9:12, 23:32] = W31 (Rows 9..11 correspond to k=3, i=0..2)
           
           return {"U": U, "V": V, "W": W}
       