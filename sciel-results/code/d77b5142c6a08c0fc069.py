"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
import math

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s kill
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        found = None
        
        # Try x values: first a few near x_min (fast solutions), then spread out
        # For small x (close to n/4), the y-range is large but we can use divisor method
        # For larger x, the y-range is smaller and linear scan is fine
        
        # Strategy: try x = x_min, x_min+1, x_min+2, x_min+5, x_min+10, x_min+20, ...
        # This covers both fast solutions (small x) and avoids getting stuck on large y-ranges
        
        x_candidates = []
        # Small offsets: likely to have solutions
        for offset in range(0, 30):
            x_candidates.append(x_min + offset)
        # Medium offsets
        for offset in [50, 100, 150, 200, 300, 500]:
            x_candidates.append(x_min + offset)
        # Remove duplicates and sort
        x_candidates = sorted(set(x_candidates))
        
        for x_attempt in x_candidates:
            if time.time() > deadline:
                break
            A = 4 * x_attempt - n
            if A <= 0:
                continue
            B = n * x_attempt
            # y must satisfy: B/A < y <= 2B/A and y > x_attempt
            y_lo = B // A + 1
            y_hi = (2 * B) // A
            if y_hi < y_lo:
                continue
            # Ensure y > x_attempt for ordering and distinctness
            if y_lo <= x_attempt:
                y_lo = x_attempt + 1
            if y_lo > y_hi:
                continue
            
            range_len = y_hi - y_lo + 1
            
            # If range is large, use divisor-based approach
            # We need (B*y) % (A*y - B) == 0
            # Let d = A*y - B, then y = (d + B)/A, so d ≡ -B (mod A)
            # And z = B*y/d = B*(d+B)/(A*d)
            # So we need A*d | B*(d+B)
            # This is complex; instead, note that d must divide B*y
            # Alternative: enumerate divisors of B and check
            
            if range_len > 1000:
                # Use divisor-based search
                # We need z = B*y / (A*y - B) to be integer
                # Let d = A*y - B > 0, so y = (d + B) / A
                # d must be a multiple of gcd(A, B) for y to be integer... 
                # Actually y = (d+B)/A must be integer, so d ≡ -B (mod A)
                # Let g = gcd(A, B). Then d = g*d', and A = g*a', B = g*b'
                # d' ≡ -b' (mod a')
                # z = B*y/d = g*b' * (d+B)/(A*d) = b' * (d+B) / (a' * d)
                # = b' * (g*d' + g*b') / (a' * g * d') = b' * (d' + b') / (a' * d')
                # So z = b' * (d' + b') / (a' * d')
                # For z to be integer, a' * d' | b' * (d' + b')
                
                A_local = A
                B_local = B
                x_local = x_attempt
                bound = DENOM_BOUND
                
                g = math.gcd(A_local, B_local)
                a_prime = A_local // g
                b_prime = B_local // g
                
                # d' must satisfy: d' ≡ -b_prime (mod a_prime)
                # and d' > 0, and y = g*(d' + b_prime) / a_prime > x_local
                # and y <= y_hi
                # and z = b_prime * (d' + b_prime) / (a_prime * d') is integer > y
                # and z <= bound
                # and all three distinct
                
                # d' = k * a_prime - b_prime for k >= 1 (to ensure d' > 0 when possible)
                # Actually d' ≡ -b_prime (mod a_prime) means d' = m * a_prime - b_prime for some m
                # We need d' > 0, so m * a_prime > b_prime, i.e., m > b_prime / a_prime
                
                m_min = b_prime // a_prime + 1
                if m_min < 1:
                    m_min = 1
                
                # y = g * (d' + b_prime) / a_prime = g * (m * a_prime - b_prime + b_prime) / a_prime = g * m
                # So y = g * m! That's much simpler.
                # z = b_prime * (d' + b_prime) / (a_prime * d') = b_prime * m * a_prime / (a_prime * (m * a_prime - b_prime))
                # = b_prime * m / (m * a_prime - b_prime)
                
                # So for each m >= m_min:
                # y = g * m
                # z = b_prime * m / (m * a_prime - b_prime)
                # Need z to be integer, z > y, z <= bound, all distinct
                
                # Check y range
                y_min_check = max(y_lo, x_local + 1)
                m_lo = (y_min_check + g - 1) // g  # ceil division
                if m_lo < m_min:
                    m_lo = m_min
                m_hi = y_hi // g
                if m_hi < m_lo:
                    continue
                
                # Limit m range to avoid too much work
                m_limit = min(m_hi, m_lo + 5000)
                
                for m in range(m_lo, m_limit + 1):
                    if (m - m_lo) % 1000 == 0 and time.time() > deadline:
                        break
                    y_val = g * m
                    if y_val <= x_local or y_val > y_hi:
                        continue
                    denom_z = m * a_prime - b_prime
                    if denom_z <= 0:
                        continue
                    num_z = b_prime * m
                    if num_z % denom_z != 0:
                        continue
                    z_val = num_z // denom_z
                    if z_val <= y_val:
                        continue
                    if z_val > bound:
                        continue
                    # Check distinctness
                    if x_local != y_val and y_val != z_val and x_local != z_val:
                        found = [x_local, y_val, z_val]
                        break
            else:
                # Linear scan for small ranges
                A_local = A
                B_local = B
                x_local = x_attempt
                bound = DENOM_BOUND
                
                for y in range(y_lo, y_hi + 1):
                    if (y - y_lo) % 1000 == 0 and time.time() > deadline:
                        break
                    denom = A_local * y - B_local
                    if denom <= 0:
                        continue
                    if (B_local * y) % denom == 0:
                        z = (B_local * y) // denom
                        if z > y and z <= bound:
                            if x_local != y and y != z and x_local != z:
                                found = [x_local, y, z]
                                break
            if found:
                break
        
        answers[i] = found
    
    return answers
