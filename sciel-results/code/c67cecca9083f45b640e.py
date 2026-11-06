import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # Deadline guard (though QR construction is instant, this is good practice)
    start_time = time.time()
    
    # Construct the extended QR(17) code
    p = 17
    
    # Quadratic residues modulo 17
    # x^2 mod 17 for x in 1..16
    # Residues: {1, 2, 4, 8, 9, 13, 15, 16}
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Create the base vector v of length 17
    # v[i] = 1 if i in residues else 0
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    mask_p = (1 << p) - 1
    for _ in range((p - 1) // 2): # 8 shifts
        basis.append(current)
        # Rotate left by 1 bit within p bits
        current = ((current << 1) | (current >> (p - 1))) & mask_p
        
    # Add the all-ones vector (9th basis vector)
    basis.append((1 << p) - 1)
    
    num_basis = len(basis) # 9
    
    # Generate all 2^9 = 512 codewords
    code = []
    
    # Pre-compute basis as a list of integers for fast XOR
    # We can use a recursive or iterative approach to generate combinations
    # Since 512 is small, a simple loop is fine and clear.
    
    # Optimization: Use bitwise operations on a list of masks
    # For each of the 512 masks, XOR the basis vectors where bits are set.
    
    for mask in range(1 << num_basis):
        word = 0
        # Iterate through basis vectors
        # Only 9 iterations per mask, 512 masks -> ~4600 ops. Instant.
        for i in range(num_basis):
            if mask & (1 << i):
                word ^= basis[i]
                
        # Compute weight of the 17-bit word
        # Using bit_count() (Python 3.10+) or bin(word).count('1')
        # Assuming Python 3.10+ as per modern standards, but fallback is safe.
        # The prompt mentions "integer XOR and population count", implying standard ops.
        weight = word.bit_count() if hasattr(int, "bit_count") else bin(word).count("1")
        
        # Determine parity bit
        # The extended QR code appends a parity bit to make the total weight even.
        # Original weight is 'weight'.
        # Parity bit = weight % 2
        parity = weight % 2
        
        # Form 18-bit integer
        # Shift original 17 bits left by 1, add parity bit at LSB
        full_word = (word << 1) | parity
        
        code.append(full_word)
        
    # The code should have 512 distinct words.
    # We can verify distinctness, but QR construction guarantees it.
    # If duplicates existed, it would be a bug in the basis generation.
    
    return code
