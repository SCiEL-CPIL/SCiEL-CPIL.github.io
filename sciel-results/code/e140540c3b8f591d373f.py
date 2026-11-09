"""
OneMax direct binary construction (100 bits)

Constructs a 100-bit binary string maximizing ones.
The optimal solution for OneMax is a string of all '1's.
"""
import time
import sys
import numpy as np

N = 100
# Deadline guard: 150s limit, safety margin 10s -> 140s
DEADLINE_SECONDS = 140.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Strategy:
    1. Check elapsed time against the deadline guard.
    2. Use NumPy to create an array of ones and convert to string.
    3. Validate the output length; if malformed, fall back to string multiplication.
    """
    start_time = time.time()
    
    try:
        # Create a NumPy array of uint8 ones
        arr = np.ones(N, dtype=np.uint8)
        
        # Convert to string of '1's
        # np.char.mod is a vectorized operation
        # However, np.char.mod('1', arr) might not work as expected for string conversion of ints directly to '1'
        # A safer way with NumPy is to convert to string array then join or use astype
        # Let's try the suggested np.char.mod first, but it usually formats numbers.
        # Actually, the directive suggests: np.char.mod('1', arr)
        # Let's test mentally: np.char.mod('1', np.array([1,1])) -> array(['1', '1'])? 
        # No, np.char.mod is for formatting. 
        # A more standard NumPy way to get '1' string for each element:
        # arr.astype(str) gives ['1', '1', ...]
        # Then join with ''
        
        # Let's follow the directive's specific hint but ensure it works.
        # If np.char.mod fails or is slow, fallback.
        # Actually, a direct string multiplication is faster. 
        # But the directive *asks* to try the NumPy path.
        
        # Let's use astype(str) and join, which is a common NumPy string conversion.
        # Or simply:
        # best_result = ''.join(np.char.mod('%d', arr)) -> This gives '111...'
        # Or even simpler:
        # best_result = np.array2string(arr, separator='', threshold=sys.maxsize, max_line_width=sys.maxsize)
        # This might add spaces.
        
        # Let's stick to the most robust NumPy string conversion:
        # arr_str = arr.astype(str)
        # best_result = ''.join(arr_str)
        
        # Let's try the specific suggestion: np.char.mod('1', arr) 
        # This is likely a misunderstanding of np.char.mod in the directive.
        # np.char.mod is a formatting function. 
        # I will use the standard astype + join method which is the standard "NumPy vectorized string conversion".
        
        arr_str = arr.astype(str)
        best_result = ''.join(arr_str)
        
        if len(best_result) != N:
            best_result = '1' * N
            
    except Exception:
        # Fallback to direct string construction if NumPy fails
        best_result = '1' * N

    # Deadline guard: 
    if time.time() - start_time > DEADLINE_SECONDS:
        return best_result
        
    return best_result
# EVOLVE-BLOCK-END
