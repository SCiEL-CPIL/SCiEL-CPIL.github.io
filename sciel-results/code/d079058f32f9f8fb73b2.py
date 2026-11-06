import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5 seconds buffer
    
    # Try to construct the 50-vertex Chung graph
    # Vertices 0..49
    n = 50
    mod = 51
    
    # Precompute quadratic residues modulo 51
    # x is a QR mod 51 iff x is a QR mod 3 and x is a QR mod 17
    # QR mod 3: {1} (0 is 0, 2 is 1)
    # QR mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    
    qr17 = {1, 2, 4, 8, 9, 13, 15, 16}
    
    def get_color(d):
        # d is in 1..50
        if d % 3 == 0:
            if d % 17 == 0:
                return 3 # Multiple of 51? No, d<51, so d=0 not possible. d=17,34 are mult of 17 but not 3.
                         # Wait, multiples of 17: 17, 34. 17%3=2, 34%3=1.
                         # So d%17==0 implies d is 17 or 34.
                         # These are NOT multiples of 3.
                         # So the condition "d%3==0" and "d%17==0" is empty for d in 1..50.
                         # Let's refine the sets.
                         # Set A: Units that are QRs
                         # Set B: Units that are Non-QRs
                         # Set C: Multiples of 3 (not 17)
                         # Set D: Multiples of 17 (not 3)
            
            # Logic correction:
            # If d is multiple of 3, it's in Set C (unless also mult of 17, which is 0)
            # If d is multiple of 17, it's in Set D
            # Else it is a unit. Check if QR.
            
            pass
        
        # Proper Logic:
        if d % 3 == 0:
            return 2 # Color 2: Multiples of 3
        if d % 17 == 0:
            return 3 # Color 3: Multiples of 17
        # d is a unit
        # Check if d is QR mod 51
        # d is QR mod 51 iff d%3 in QR3 and d%17 in QR17
        # QR3 = {1}
        # QR17 = qr17
        if d % 3 == 1 and d % 17 in qr17:
            return 0 # Color 0: QR
        else:
            return 1 # Color 1: Non-QR

    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (i - j) % mod
            # d will be 1..50
            colors.append(get_color(d))
    
    # Verify
    # Check for monochromatic triangles
    # This is O(n^3) which is 20k ops, very fast.
    is_valid = True
    for i in range(n):
        for j in range(i + 1, n):
            c_ij = colors[i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)] # Indexing is tricky, let's use a helper or list of lists
            # Actually, let's just use a 2D matrix for verification to be safe and simple, though it uses more memory.
            # n=50, 50x50 matrix is tiny.
            
    # Re-verification with matrix for clarity
    mat = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j] = colors[idx]
            mat[j][i] = colors[idx]
            idx += 1
            
    for i in range(n):
        if time.time() > deadline:
            # Fallback to seed solution if verification takes too long (unlikely)
            return {'n': 32, 'colors': [0]*496} # Dummy fallback, should not happen
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if mat[i][j] == mat[j][k] == mat[i][k]:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
            
    if is_valid:
        return {'n': 50, 'colors': colors}
    else:
        # If the 50-vertex construction fails, return the 32-vertex seed solution
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors_seed = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors_seed}
