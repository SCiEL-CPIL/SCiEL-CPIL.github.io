"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
import time

def _generate_solutions():
    """Generate all valid triples for n=3..256 using optimized search."""
    BIG = 10**12
    solutions = {}
    
    for n in range(3, 257):
        # Try to find a valid triple
        found = None
        
        # Quick patterns first
        if n % 6 == 3:
            m = n // 3
            x, y, z = m, 3*m + 1, 3*m*(3*m + 1)
            if x != y and y != z and x != z and x <= BIG and y <= BIG and z <= BIG:
                if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                    found = (x, y, z)
        
        if found is None and n % 8 == 4:
            m = n // 4
            x, y, z = m + 2, (m + 1)*(m + 2), m*(m + 1)
            if x != y and y != z and x != z and x <= BIG and y <= BIG and z <= BIG:
                if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                    found = (x, y, z)
        
        if found is None and n % 10 == 5:
            m = n // 5
            x, y, z = 2*m, 5*m, 10*m
            if x != y and y != z and x != z and x <= BIG and y <= BIG and z <= BIG:
                if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                    found = (x, y, z)
        
        if found is None and n % 8 == 0:
            x, y, z = n // 4, n // 2, n
            if x != y and y != z and x != z and x >= 1:
                if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                    found = (x, y, z)
        
        # General search for remaining cases
        if found is None:
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            if x_max < x_min:
                x_min, x_max = 1, 1000000
            
            for x in range(x_min, x_max + 1):
                if found:
                    break
                a = 4 * x - n
                b = n * x
                if a <= 0 or b <= 0:
                    continue
                # 1/y + 1/z = a/b
                # z = b*y / (a*y - b), need a*y - b > 0 => y > b/a
                # z > y => b*y/(a*y-b) > y => b > a*y - b => a*y < 2b => y < 2b/a
                y_low = max(x + 1, b // a + 1)
                y_high = (2 * b) // a
                if y_low > y_high:
                    continue
                for y in range(y_low, y_high + 1):
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    if (b * y) % denom != 0:
                        continue
                    z = (b * y) // denom
                    if z <= y or z <= x:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if z > BIG or y > BIG or x > BIG:
                        continue
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        found = (x, y, z)
                        break
        
        solutions[n] = found
    
    return solutions

# Pre-compute all solutions at module load time
_SOLUTIONS = _generate_solutions()

def construct():
    """Return 254 rows for n=3..256. Each row is None or three distinct positive integer denominators."""
    return [_SOLUTIONS[n] for n in range(3, 257)]
