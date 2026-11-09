# EVOLVE-BLOCK-START
def construct():
    # Create a NumPy unsigned integer array of length N, initialized to 1
    arr = np.ones(N, dtype=np.uint8)
    # Convert the integer array to single-character Unicode strings ('1')
    str_arr = arr.astype('U1')
    # Join the array elements into a single string
    return ''.join(str_arr)
# EVOLVE-BLOCK-END
