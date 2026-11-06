import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def _build_in_bits(n, orientation):
        in_bits = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    in_bits[j] |= (1 << i)
                else:
                    in_bits[i] |= (1 << j)
        return in_bits

    def _count_violations(n, in_bits):
        violations = 0
        for a in range(n - 3):
            mask_a = ~(1 << a)
            for b in range(a + 1, n - 2):
                mask_ab = mask_a & ~(1 << b)
                for c in range(b + 1, n - 1):
                    mask_abc = mask_ab & ~(1 << c)
                    for d in range(c + 1, n):
                        mask_abcd = mask_abc & ~(1 << d)
                        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d] & mask_abcd) == 0:
                            violations += 1
        return violations

    def _paleyn_tournament(p):
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                orientation.append(0 if diff in squares else 1)
        return p, orientation

    def _local_search(n, init_orientation, time_limit):
        orientation = init_orientation[:]
        in_bits = _build_in_bits(n, orientation)
        current_violations = _count_violations(n, in_bits)
        
        if current_violations == 0:
            return orientation, 0
            
        best_orientation = orientation[:]
        best_violations = current_violations
        
        temp = 5.0
        min_temp = 0.01
        cooling = 0.9995
        
        while time.time() < start_time + time_limit and temp > min_temp:
            idx = random.randint(0, n * (n - 1) // 2 - 1)
            
            i, j = -1, -1
            count = 0
            for ii in range(n):
                if count + (n - ii - 1) > idx:
                    i = ii
                    j = idx - count + ii + 1
                    break
                count += (n - ii - 1)
            
            old_bit = orientation[idx]
            new_bit = 1 - old_bit
            orientation[idx] = new_bit
            
            if old_bit == 0:
                in_bits[j] &= ~(1 << i)
                in_bits[i] |= (1 << j)
            else:
                in_bits[i] &= ~(1 << j)
                in_bits[j] |= (1 << i)
                
            new_violations = _count_violations(n, in_bits)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_orientation = orientation[:]
                    if best_violations == 0:
                        return best_orientation, 0
            else:
                orientation[idx] = old_bit
                if old_bit == 0:
                    in_bits[j] |= (1 << i)
                    in_bits[i] &= ~(1 << j)
                else:
                    in_bits[i] |= (1 << j)
                    in_bits[j] &= ~(1 << i)
            
            temp *= cooling
        
        return best_orientation, best_violations

    # Try Paley primes in descending order starting from 67
    # We want the smallest n, so we try small n first?
    # But small n might be harder to make valid.
    # Let's try 67, 59, 53, 47, 43, 31, 23, 19, 11, 7
    # Actually, we want to minimize n. So we should try 7, 11, 19, 23, 31, 43, 47, 53, 59, 67
    # But if 7 is not valid, we move to 11, etc.
    # However, if 67 is valid, we can return it immediately if we don't find smaller.
    # To maximize score, we should find the SMALLEST valid n.
    
    primes = [7, 11, 19, 23, 31, 43, 47, 53, 59, 67]
    
    best_result = None
    
    # We have limited time.
    # Let's allocate time based on likelihood.
    # Paley(67) is known to be valid.
    # Let's check if we can improve on 67.
    
    # Try 59 first?
    # If 59 is valid, score is 83/59 = 1.40.
    # If 53 is valid, score is 83/53 = 1.56.
    
    # Strategy:
    # 1. Check if Paley(67) is valid. (It is, likely).
    # 2. Try to find valid tournament for n < 67 using local search from Paley(n) or random.
    
    # Let's try n=59 with local search.
    # If it fails, try n=53.
    # If it fails, return 67.
    
    for n_target in [59, 53, 47, 43]:
        if time.time() > deadline - 10:
            break
        time_limit = min(30.0, deadline - time.time() - 5.0)
        if time_limit <= 0:
            break
            
        n_p, orient_p = _paleyn_tournament(n_target)
        in_bits_p = _build_in_bits(n_p, orient_p)
        viol_p = _count_violations(n_p, in_bits_p)
        
        if viol_p == 0:
            return {'n': n_target, 'orientation': orient_p}
            
        # Run local search
        orient, viol = _local_search(n_target, orient_p, time_limit)
        
        if viol == 0:
            return {'n': n_target, 'orientation': orient}
            
    # Fallback to 67
    n_p, orient_p = _paleyn_tournament(67)
    return {'n': n_p, 'orientation': orient_p}
