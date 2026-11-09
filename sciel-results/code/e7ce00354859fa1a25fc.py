"""OneMax 100-bit binary construction: maximize number of ones."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is all ones: "1" * 100.
    
    This implementation follows the directive to use a NumPy boolean array 
    initialized to True, then converts it to a string of '1's.
    This leverages vectorized memory allocation and conversion, representing 
    a distinct numerical pathway to the same optimal result.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # 1. Initialize a NumPy boolean array of length N with all True values.
    #    In the context of OneMax, True represents a '1' bit.
    bits = np.ones(N, dtype=bool)
    
    # 2. Convert the boolean array to a string of '1's.
    #    astype('U1') converts the boolean array to a unicode string array of length 1,
    #    where True becomes 'T' or '1' depending on representation, but typically 
    #    we want explicit '1's. 
    #    Actually, np.array([True]).astype('U1') gives 'T'. 
    #    We need '1'. So we can use np.where(bits, '1', '0') or simply map.
    #    However, the directive says: "Convert this boolean array directly to a binary string using astype('U1')"
    #    Let's check what astype('U1') does to bools.
    #    np.array([True, False]).astype('U1') -> array(['T', 'F'], dtype='<U1')
    #    This yields 'T' and 'F', not '1' and '0'.
    #    The directive might be slightly imprecise about the exact output of astype('U1') for bools 
    #    or expects us to handle the mapping. 
    #    To be safe and ensure the output is strictly '1's, I will use a method that guarantees '1'.
    #    But I must follow the "spirit" of using the boolean array.
    #    Let's use np.char.mod or simple concatenation of the mapped values.
    #    Or, more simply, since we know they are all True, we can just return "1"*N.
    #    But to respect the "distinct numerical representation pathway", I will perform the conversion.
    
    # Map True to '1' and False to '0' using vectorized operations
    # np.where(bits, '1', '0') returns an array of strings
    char_array = np.where(bits, '1', '0')
    
    # Join the array into a single string
    result = ''.join(char_array)
    
    # Check deadline
    if time.time() > deadline:
         # Fallback: though O(N) is fast, if we were stuck, return valid optimal solution
         return "1" * N
         
    return result
# EVOLVE-BLOCK-END
