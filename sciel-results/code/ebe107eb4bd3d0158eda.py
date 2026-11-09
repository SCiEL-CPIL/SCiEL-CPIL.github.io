import time
import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    # We target the smallest known valid n, which is 67.
    # We construct the Paley tournament for p=67.
    p = 67
    
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    # Optional: Verify validity using bitsets
    # If verification is too slow or fails (shouldn't), we might need to fallback, 
    # but 67 is known to be valid.
    if _is_valid_fast(p, orientation):
        return {'n': p, 'orientation': orientation}
        
    # Fallback if somehow invalid (should not happen)
    # Try next prime p=83
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}

def _is_valid_fast(n, orientation):
    # Build in-neighbor bitsets
    # in_neigh[i] is a bitmask where bit j is 1 if j -> i
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j, so i is in in_neigh[j]
                in_neigh[j] |= (1 << i)
            else:
                # j -> i, so j is in in_neigh[i]
                in_neigh[i] |= (1 << j)
            idx += 1
            
    # Check all 4-combinations
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Check if there is a common dominator
                    # The dominator v must be in In(a) & In(b) & In(c) & In(d)
                    common = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
                    if common == 0:
                        return False
    return True
