def _get_full_tensor_matrix(U, V, W, n, m, p):
    """
    Returns a matrix of shape (n*m*p*p*n? No, flattened tensor size, rank).
    Flattened index: ((i*m+j) * (m*p) + (j*p+k)) * (p*n) + (k*n+i)
    This is too big.
    Instead, we just need to check linear independence of the RANK-1 tensors.
    """
    pass
