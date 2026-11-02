def get_u_indices(rows, cols):
    """Get the 16-length index vector for a 2x2 submatrix."""
    # rows = [r0, r1], cols = [c0, c1]
    # Position (0,0) → r0*4+c0, Position (0,1) → r0*4+c1
    # Position (1,0) → r1*4+c0, Position (1,1) → r1*4+c1
    return np.array([rows[0]*4+cols[0], rows[0]*4+cols[1], 
                     rows[1]*4+cols[0], rows[1]*4+cols[1]])

def get_v_indices(rows, cols):
    """Get the index vector for a 2x2 submatrix of B (m*p matrix)."""
    # For 4x4x4: B is 4x4, V index = j*4+k
    # rows = [r0, r1] (j indices), cols = [c0, c1] (k indices)
    return np.array([rows[0]*4+cols[0], rows[0]*4+cols[1],
                     rows[1]*4+cols[0], rows[1]*4+cols[1]])

def get_w_indices(rows, cols):
    """Get the index vector for a 2x2 submatrix of C (n*p matrix)."""
    # For 4x4x4: C is 4x4, W index = k*4+i
    # rows = [k0, k1] (k indices), cols = [i0, i1] (i indices)
    return np.array([rows[0]*4+cols[0], rows[0]*4+cols[1],
                     rows[1]*4+cols[0], rows[1]*4+cols[1]])
