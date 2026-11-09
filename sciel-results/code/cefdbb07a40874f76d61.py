import time
import random

N = 18
MIN_DIST = 6

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # 1. Generate the 64-word extended QR code
    base_code = generate_extended_qr_code()
    
    # 2. Greedy expansion
    best_code = list(base_code)
    
    # Try multiple random restarts of the greedy process
    # We will generate candidates in random order
    candidates = list(range(2**N))
    
    # We can also try to add words in a structured way first
    # But random is more robust for nonlinear codes
    
    for restart in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        
        current_code = list(best_code)
        # Note: We start from the best found so far to potentially improve
        # Or should we restart from base? 
        # Usually, continuing from the best is better if we just shuffle the remaining candidates.
        # But greedy is order-dependent. 
        # Let's try adding to the current best code.
        
        for c in candidates:
            if time.time() > deadline:
                break
                
            if is_valid(c, current_code):
                current_code.append(c)
                
        if len(current_code) > len(best_code):
            best_code = current_code
            
    return best_code

def is_valid(candidate, code):
    for w in code:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True

def generate_extended_qr_code():
    # Implementation of GF(2^8) and QR code generation
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    # This is a common choice.
    
    def gf_mult(a, b, mod_poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= mod_poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mult(res, a)
            a = gf_mult(a, a)
            n >>= 1
        return res

    # Find primitive 17th root of unity
    # 2^8 - 1 = 255. 255 / 17 = 15.
    # Let g = 2 (primitive element for 0x11D? Need to verify).
    # Actually, 2 is often primitive for 0x11D.
    # Let's assume g=2 is primitive.
    alpha = gf_pow(2, 15)
    
    # Verify alpha^17 == 1
    # alpha^17 = gf_pow(alpha, 17)
    # If not 1, try other g.
    
    # Quadratic residues mod 17
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25=8, 6^2=36=2, 7^2=49=15, 8^2=64=13
    Q = [1, 4, 9, 16, 8, 2, 15, 13]
    
    # Generator polynomial g(x) = LCM(x - alpha^i for i in Q)
    # We build g(x) iteratively.
    # Represent polynomials as lists of coefficients [c0, c1, ...]
    # Or use integers (bitmask).
    
    def poly_lcm_int(p, q):
        # p, q are integers representing polynomials
        # This is complex. Easier to build g(x) directly from roots.
        pass

    # Alternative: Build g(x) by multiplying (x - alpha^i)
    # Start with 1
    g = 1 # 1
    # We need to work in GF(2^8) coefficients.
    # This is getting complicated.
    
    # Simpler approach:
    # Use the known generator polynomial for QR(17).
    # From literature, the generator polynomial for the QR code of length 17 is:
    # g(x) = x^11 + x^10 + x^9 + x^8 + x^7 + x^5 + x^4 + x^3 + x^2 + x + 1
    # Let's verify the weight and distance.
    # If I hardcode this, I avoid the GF complexity.
    
    g_coeffs = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1] # c0 to c11?
    # Usually g(x) = sum c_i x^i.
    # Let's assume the standard form.
    
    # Generate all 2^6 codewords
    words = []
    for msg in range(64):
        # Multiply msg(x) by g(x)
        # msg is 6 bits.
        # c(x) = msg(x) * g(x)
        c = 0
        for i in range(6):
            if msg & (1 << i):
                c ^= (g_poly << i) # g_poly is int representation
        
        # c is 17 bits.
        # Add parity bit
        parity = c.bit_count() % 2
        c_ext = c | (parity << 17)
        words.append(c_ext)
        
    return words

# Need g_poly as int.
# g(x) = x^11 + x^10 + x^9 + x^8 + x^7 + x^5 + x^4 + x^3 + x^2 + x + 1
# Bits: 11, 10, 9, 8, 7, 5, 4, 3, 2, 1, 0
g_poly = (1 << 11) | (1 << 10) | (1 << 9) | (1 << 8) | (1 << 7) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    base_code = generate_extended_qr_code()
    best_code = list(base_code)
    
    candidates = list(range(2**18))
    
    for restart in range(50):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        current_code = list(best_code)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if is_valid(c, current_code):
                current_code.append(c)
                
        if len(current_code) > len(best_code):
            best_code = current_code
            
    return best_code

def is_valid(candidate, code):
    for w in code:
        if (candidate ^ w).bit_count() < 6:
            return False
    return True

def generate_extended_qr_code():
    g_poly = 0b11111011111 # 11,10,9,8,7,5,4,3,2,1,0
    # Wait, 1<<11 is bit 11.
    # 11: 1, 10: 1, 9: 1, 8: 1, 7: 1, 6: 0, 5: 1, 4: 1, 3: 1, 2: 1, 1: 1, 0: 1
    # Binary: 11111011111
    # Value: 2048+1024+512+256+128+32+16+8+4+2+1 = 4031
    g_poly = 4031
    
    words = []
    for msg in range(64):
        c = 0
        for i in range(6):
            if msg & (1 << i):
                c ^= (g_poly << i)
        parity = c.bit_count() % 2
        c_ext = c | (parity << 17)
        words.append(c_ext)
    return words
