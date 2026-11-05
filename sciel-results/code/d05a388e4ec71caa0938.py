    # Create an integer with all N bits set to 1 using bitwise OR
    mask = (1 << N) - 1
    val = 0 | mask
    
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(val)[2:]
