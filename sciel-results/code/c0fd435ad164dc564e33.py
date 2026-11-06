"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct distinct three-term unit-fraction representations of 4/n for n=3..256.
    Uses algebraic parameterization: 1/x + 1/y + 1/z = 4/n
    with x = (n+k)/4, y = (n+k)/2, z = 4n(n+k)/(k(4k-n))
    for integer k satisfying constraints.
    """
    answers = []
    
    for n in range(3, 257):
        found = None
        
        # Try the algebraic identity approach:
        # Let x = (n+k)/4, y = (n+k)/2
        # Then 1/x + 1/y = 4/(n+k) + 2/(n+k) = 6/(n+k)
        # We need 1/z = 4/n - 6/(n+k) = (4(n+k) - 6n)/(n(n+k)) = (4k - 2n)/(n(n+k))
        # So z = n(n+k)/(4k - 2n) = n(n+k)/(2(2k-n))
        
        # For z to be positive: 2k - n > 0 => k > n/2
        # For x, y to be integers: n+k must be divisible by 4 (for x) and by 2 (for y, which is implied by /4)
        # So n+k ≡ 0 (mod 4), i.e., k ≡ -n (mod 4)
        
        # Also need x, y, z to be distinct positive integers <= 10^12
        
        # Let's iterate over possible k values where n+k is divisible by 4
        # k must be > n/2 and k ≡ -n (mod 4)
        
        # Start from smallest valid k
        start_k = (n // 2) + 1
        # Adjust to next k such that (n + k) % 4 == 0
        remainder = (n + start_k) % 4
        if remainder != 0:
            start_k += (4 - remainder) % 4
        if (n + start_k) % 4 != 0:
            start_k += 4
        
        # Try a range of k values
        for k in range(start_k, start_k + 200):
            if (n + k) % 4 != 0:
                continue
            
            x = (n + k) // 4
            y = (n + k) // 2
            
            # Check distinctness
            if x == y:
                continue
            
            # z = n(n+k)/(2(2k-n))
            denom_z = 2 * (2 * k - n)
            if denom_z <= 0:
                continue
            
            num_z = n * (n + k)
            if num_z % denom_z != 0:
                continue
            
            z = num_z // denom_z
            
            if z <= 0:
                continue
            
            # Check all distinct
            if x == y or y == z or x == z:
                continue
            
            # Check bounds
            if x > 10**12 or y > 10**12 or z > 10**12:
                continue
            
            # Verify the equation by cross multiplication
            # 1/x + 1/y + 1/z = 4/n
            # (yz + xz + xy) / (xyz) = 4/n
            # n*(yz + xz + xy) = 4*xyz
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                found = [x, y, z]
                break
        
        # If the algebraic approach didn't work, try a more general search
        if found is None:
            # General approach: fix x, then solve for y, z
            # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            # Let A = 4x - n, B = nx
            # 1/y + 1/z = A/B
            # (y+z)/(yz) = A/B => B(y+z) = A*y*z => A*y*z - B*y - B*z = 0
            # (Ay - B)(Az - B) = B^2
            # So we need factors of B^2
            
            # Try x values starting from ceil(n/4) + 1 to ensure 4x - n > 0
            min_x = (n + 3) // 4 + 1
            max_x = min(10**12, 4 * n * 10**6)  # Reasonable upper bound
            
            found = None
            for x in range(min_x, min(min_x + 500, max_x + 1)):
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                # We need (Ay - B)(Az - B) = B^2
                # Let u = Ay - B, v = Az - B, so u*v = B^2
                # y = (u + B)/A, z = (v + B)/A
                # u and v must be positive divisors of B^2 such that u*v = B^2
                # and (u+B) % A == 0 and (v+B) % A == 0
                # and y > x, z > y (or some ordering)
                
                # Factor B^2 is expensive, so let's try a simpler approach:
                # Iterate over y and check if z is integer
                # 1/z = A/B - 1/y = (Ay - B)/(By)
                # z = By/(Ay - B)
                # Need Ay - B > 0 => y > B/A = nx/(4x-n)
                # And Ay - B divides By
                
                min_y = B // A + 1
                if min_y <= x:
                    min_y = x + 1
                
                # Upper bound for y: as y increases, z decreases
                # We want z > y, so By/(Ay-B) > y => B/(Ay-B) > 1 => B > Ay - B => 2B > Ay => y < 2B/A
                max_y = min(2 * B // A + 1, min_y + 200)
                
                for y in range(min_y, max_y):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom != 0:
                        continue
                    z = (B * y) // denom
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    # Verify
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                if found is not None:
                    break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
