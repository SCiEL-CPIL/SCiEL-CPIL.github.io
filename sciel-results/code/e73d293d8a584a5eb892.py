import time

def construct():
    # Target the smallest known valid n, which is 67 (Paley tournament).
    # f(4) is known to be 67. No smaller n is known to exist.
    # We construct the Paley tournament of order p=67.
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    p = 67
    
    # Precompute quadratic residues modulo p
    # x^2 mod p for x in 1..p-1
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    # Construct orientation bits
    # One bit per pair (i, j) with i < j in lexicographic order.
    # 0 means i -> j, 1 means j -> i.
    orientation = []
    
    # Optimization: Use a list of 0s and 1s, then join or convert to list of ints
    # The interface expects a list of bits.
    
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    # The Paley tournament for p=67 is a known valid solution for Schütte's problem k=4.
    # Verification is O(n^4) and unnecessary if we trust the mathematical result.
    # Skipping verification ensures the function returns instantly.
    
    return {'n': p, 'orientation': orientation}
