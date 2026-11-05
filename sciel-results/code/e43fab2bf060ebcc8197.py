import time
import itertools

def construct():
    start_time = time.time()
    timeout = 50.0
    
    best_count = float('inf')
    best_coeffs = [1]*12
    
    # Brute force signs for 12 terms
    # c_i in {-1, 1}
    # There are 2^12 = 4096 patterns.
    
    for bits in range(1 << 12):
        if time.time() - start_time > timeout:
            break
            
        coeffs = []
        for i in range(12):
            if (bits >> i) & 1:
                coeffs.append(1)
            else:
                coeffs.append(-1)
        
        # Compute square
        # Square length is 2*12 - 1 = 23
        square = [0]*23
        for i in range(12):
            for j in range(12):
                square[i+j] += coeffs[i] * coeffs[j]
                
        count = sum(1 for x in square if x != 0)
        
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            
    # If best_count is still high, try small integers?
    # For now, return best_coeffs padded to length 12 (or 41?)
    # The task says "Return coefficient list in ascending degree, length<=41"
    # My coeffs are for exponents 0..11.
    # I should return a list of length 12.
    
    return best_coeffs
