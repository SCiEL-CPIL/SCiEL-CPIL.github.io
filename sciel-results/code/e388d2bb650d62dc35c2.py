import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values.
    # Each True represents a '1' in the binary string.
    bits = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's.
    # Since all are True, we can just join '1' N times.
    # To strictly follow the "convert array to string" idea while being efficient:
    # np.char.mod is overkill. 
    # A direct conversion from bool array to string of '1's and '0's would be:
    # ''.join(['1' if b else '0' for b in bits])
    # But since we know they are all 1, we can be simpler or stick to the vectorized spirit.
    # Let's use the vectorized conversion to be safe and general, though for all-ones it's trivial.
    # Actually, the directive says "convert it to a string of '1's".
    # If we assume the array is all True, the string is just "1" * N.
    # But to respect the "array-based construction" path:
    
    # Convert bool to int, then to string
    # This is slightly slower than "1"*N but follows the directive's structural hint.
    # However, for N=100, the difference is negligible.
    # Let's just return the string of ones, constructed via the array logic if we were general, 
    # but here we know the optimal is all ones.
    
    # The most robust "array-based" way to get the string from the array:
    # int_array = bits.astype(int)
    # return ''.join(map(str, int_array))
    
    # Given the task is specifically OneMax and we want the max score, 
    # and the directive asks for this specific strategy:
    return ''.join('1' for _ in bits)
# EVOLVE-BLOCK-END
