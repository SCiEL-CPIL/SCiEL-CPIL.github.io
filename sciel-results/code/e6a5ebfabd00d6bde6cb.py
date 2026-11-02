import numpy as np

D = 11

def construct():
    # Generate all vectors with exactly two non-zero entries, each +/- 1
    # This corresponds to the set S_2 which is a valid kissing configuration
    # with score 220.
    
    # Indices for the two non-zero positions
    i_indices, j_indices = np.meshgrid(np.arange(D), np.arange(D), indexing='ij')
    mask = i_indices < j_indices
    i_list = i_indices[mask]
    j_list = j_indices[mask]
    
    # Signs for the two positions
    signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int64)
    
    vectors = []
    for i, j in zip(i_list, j_list):
        for s in signs:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s[0]
            v[j] = s[1]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
