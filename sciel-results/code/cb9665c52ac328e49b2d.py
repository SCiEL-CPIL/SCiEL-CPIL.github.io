import time

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break

        found = None
        # x must be > n/4 and < 3n/4 for x < y < z
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            # Check time occasionally
            if time.time() > deadline:
                return results
            
            denom_x = 4 * x - n
            if denom_x <= 0:
                continue
            
            # y > nx / (4x - n)
            # y < 2nx / (4x - n)
            y_min_val = (n * x) // denom_x
            y_max_val = (2 * n * x) // denom_x
            
            # y must be > x
            y_start = max(x + 1, y_min_val + 1)
            y_end = y_max_val
            
            if y_start > y_end:
                continue
            
            for y in range(y_start, y_end + 1):
                # Calculate numerator and denominator for z
                # 1/z = (4xy - nx - ny) / (nxy)
                num_z = 4 * x * y - n * x - n * y
                if num_z <= 0:
                    continue
                
                den_z = n * x * y
                
                if den_z % num_z == 0:
                    z = den_z // num_z
                    # Check distinctness and bounds
                    # We enforced x < y. Need z > y.
                    if z > y and z <= DENOM_BOUND:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        results[idx] = found

    return results
