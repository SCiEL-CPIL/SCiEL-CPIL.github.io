"""Combined program for distinct three-term unit-fraction reps of 4/n."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # Safety margin under 90s kill
    results = [None] * 254
    BOUND = 10**12
    
    # Iterate over n from 3 to 256
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # x must satisfy: n/4 < x < 3n/4
        # So x_min = floor(n/4) + 1
        # x_max = ceil(3n/4) - 1 (conservative, usually (3n-1)//4 works)
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        # If x_min > x_max, no solution for this n? 
        # Actually, for n=3: x_min=1, x_max=1. 
        # For n=4: x_min=2, x_max=2.
        # The range is small.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # => (A*y - B)(A*z - B) = B^2
            # Or directly: z = B*y / (A*y - B)
            
            # Constraints on y:
            # 1. A*y - B > 0  => y > B/A
            # 2. z > y        => B*y / (A*y - B) > y
            #                   => B > A*y - B
            #                   => 2B > A*y
            #                   => y < 2B/A
            # 3. y > x (for distinctness and ordering x < y < z)
            
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A
            
            if y_max < y_min:
                continue
                
            # Iterate y in the small range [y_min, y_max]
            # This range is typically very small for the first few x
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                    
                num = B * y
                if num % denom == 0:
                    z = num // denom
                    # Check distinctness and bounds
                    if z > y and z <= BOUND:
                        if x != y and y != z and x != z:
                            found = [x, y, z]
                            break
                            
            if found:
                break
                
        results[idx] = found
        
    return results
# EVOLVE-BLOCK-END
