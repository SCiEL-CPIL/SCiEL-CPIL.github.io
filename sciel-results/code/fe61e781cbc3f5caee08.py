"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses NumPy vectorization to create the optimal string of all ones.
    """
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to simple string multiplication if we are close to timeout
        # though this shouldn't happen for a direct construction
        return "1" * N

    # Create a NumPy array of uint8 filled with 1s
    arr = np.ones(N, dtype=np.uint8)
    
    # Convert each element to its string representation and join
    # Using np.char.mod for vectorized string formatting as per directive
    # Note: np.char.mod('1', arr) is slightly odd, '1' is a format string.
    # Actually, np.char.mod expects a format string and values.
    # '1' is not a valid format specifier for numbers in the standard sense 
    # (like %d), but let's check how numpy handles it.
    # Actually, np.char.mod is for formatting. 
    # A safer and more standard vectorized way to join is ''.join(arr.astype(str))
    # However, the directive specifically says: "cast it to a string using np.char.mod('1', arr)"
    # Let's verify if np.char.mod('1', arr) works as intended.
    # If arr is [1, 1], np.char.mod('1', arr) might produce ['1', '1']?
    # Actually, '1' is just a string. It doesn't have a format placeholder like '%d'.
    # If I run np.char.mod('1', [1,1]), it might error or behave unexpectedly.
    # Let's look at the example in the prompt's "Strategy to try":
    # "cast it to a string using np.char.mod('1', arr)"
    # This might be a typo in the directive or a specific numpy feature I'm misremembering.
    # np.char.mod is typically used like np.char.mod('%d', arr).
    # If I use np.char.mod('1', arr), it might just return '1' for each element?
    # Let's test mentally: np.char.mod('1', 1) -> '1'?
    # If the directive is strict, I should try it. But if it fails, the program crashes.
    # A safer interpretation of "vectorized string formatting" is arr.astype(str).
    # However, the prompt says: "Strategy to try in this change: ... using np.char.mod('1', arr)"
    # Let's assume the directive implies a specific call. 
    # If I look at the observed change p47, it used:
    # result = np.char.mod('1', arr)
    # and it scored 2.0. So it works.
    # It seems np.char.mod treats the first arg as a template. If the template is '1', 
    # and there are no placeholders, it might just repeat '1'? Or maybe it's interpreted as a format?
    # Actually, looking at numpy docs, np.char.mod is for formatting.
    # If I pass '1' and an array, it might broadcast?
    # Given p47 succeeded, I will replicate that logic exactly.
    
    result = np.char.mod('1', arr)
    
    # Join the array of strings into a single string
    return ''.join(result)
# EVOLVE-BLOCK-END
