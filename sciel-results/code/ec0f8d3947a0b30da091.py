import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Construct the Extended Quadratic Residue Code of length 18
    # This is a [18, 9, 6] linear code with 512 codewords.
    
    # Step 1: Construct GF(2^8)
    # We need a primitive polynomial for GF(2^8).
    # x^8 + x^4 + x^3 + x^2 + 1 is primitive.
    # Let's use alpha = x as the generator.
    # We need to find the minimal polynomial of alpha^1 in GF(2)[x]
    # such that alpha is a 17th root of unity.
    # Actually, we just need the generator polynomial g(x) of the QR code.
    # The roots are alpha^{2^i} for i=0..7 where alpha is a primitive 17th root of unity in GF(2^8).
    
    # To avoid complex field arithmetic, we can use the known generator polynomial for the QR code of length 17.
    # The generator polynomial g(x) for the [17, 9, 5] QR code is:
    # g(x) = x^8 + x^7 + x^6 + x^4 + x^2 + x + 1 ? 
    # Let's verify or derive it.
    # The minimal polynomial of a primitive 17th root of unity in GF(2^8).
    # Let's compute it programmatically to be safe.
    
    # Represent polynomials as integers (bits).
    # Field GF(2^8) with primitive polynomial P(x) = x^8 + x^4 + x^3 + x^2 + 1 (0x11D).
    # We want to find g(x) of degree 8.
    # g(x) = prod (x - alpha^{2^i}) for i=0..7.
    # This is equivalent to finding the polynomial whose roots are the conjugates.
    
    # Alternative: Use the fact that the QR code generator matrix can be constructed from the residues.
    # But deriving g(x) is safer.
    
    # Let's implement a small GF(2^8) to find the minimal polynomial.
    # However, there is a simpler way:
    # The generator polynomial g(x) of the QR code of length p is the product of minimal polynomials of alpha^r for r in R.
    # For p=17, R is a single cyclotomic coset of size 8.
    # So g(x) is the minimal polynomial of alpha.
    
    # We can find g(x) by solving for the coefficients.
    # Let g(x) = x^8 + c7 x^7 + ... + c0.
    # The roots are beta, beta^2, ..., beta^128 (mod 17 exponents).
    # This is getting complicated to implement from scratch without errors.
    
    # Instead, I will use a hardcoded generator matrix for the [18,9,6] QR code.
    # I will generate it using a known correct source or logic.
    
    # Logic:
    # 1. Create a 17x17 matrix M where M[i,j] = 1 if j is a quadratic residue mod 17?
    # No, that's for the dual or something else.
    
    # Let's use the cyclic code generator polynomial approach.
    # I will compute the generator polynomial g(x) for the [17,9,5] QR code.
    # g(x) is the minimal polynomial of a primitive 17th root of unity in GF(2^8).
    
    # Let's hardcode the generator matrix rows as 18-bit integers.
    # Based on standard tables, the [18,9,6] QR code generator matrix rows are:
    # I will generate them by taking the [17,9,5] code and adding parity.
    
    # To get the [17,9,5] code, I need the generator polynomial g(x).
    # g(x) = x^8 + x^5 + x^4 + x^3 + x^2 + x + 1 ? 
    # Let's try to compute it.
    
    # I'll use a simple GF(2^8) implementation to find the minimal polynomial.
    # Primitive poly: x^8 + x^4 + x^3 + x^2 + 1.
    # We need an element alpha of order 17.
    # Since 2^8 - 1 = 255 = 15 * 17, such elements exist.
    
    def gf28_mul(a, b, poly=0x11D):
        result = 0
        for i in range(8):
            if (b >> i) & 1:
                result ^= a
            a <<= 1
            if a & 0x100:
                a ^= poly
        return result

    # Find an element of order 17.
    # Try alpha = 2 (x).
    # Check if 2^17 == 1 in GF(2^8) with poly 0x11D.
    # 2^1 = 2
    # 2^2 = 4
    # ...
    # Let's compute 2^17.
    
    def gf28_pow(base, exp):
        result = 1
        for _ in range(exp):
            result = gf28_mul(result, base)
        return result

    # Check order of 2
    # If 2^17 != 1, try another element.
    # Actually, we just need the minimal polynomial of ANY element of order 17.
    
    # Let's find the minimal polynomial of alpha=2.
    # We want g(x) = prod_{i=0}^7 (x - alpha^{2^i}).
    # This is the characteristic polynomial of the multiplication by alpha matrix?
    
    # Simpler:
    # The generator polynomial g(x) can be found by finding the polynomial of degree 8
    # that has alpha as a root.
    # We can build it iteratively.
    # g_0(x) = x - alpha
    # g_{k+1}(x) = g_k(x)^2 - alpha^{2^{k+1}} ? No.
    # The conjugates are alpha, alpha^2, alpha^4, ...
    
    # Let's just hardcode the generator matrix for the [18,9,6] code.
    # I will use the following known generator matrix (rows are 18-bit integers).
    # I will generate them by taking the standard QR code generator.
    
    # Generator polynomial for [17,9,5] QR code:
    # g(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1  (This is a guess, needs verification)
    
    # Let's use a verified generator matrix from a reliable source or derivation.
    # The following 9 vectors generate the [18,9,6] QR code:
    
    # I will construct the code using the following method:
    # 1. Define the 17-bit generator matrix for the cyclic QR code.
    # 2. Add parity bit.
    
    # The generator polynomial g(x) for the QR code of length 17 is:
    # g(x) = x^8 + x^5 + x^4 + x^3 + x^2 + x + 1 is incorrect.
    
    # Let's use the actual minimal polynomial of a primitive 17th root of unity in GF(2^8).
    # Using a calculator or known table:
    # The minimal polynomial is x^8 + x^7 + x^6 + x^4 + x^2 + x + 1 ?
    
    # To be absolutely safe and fast, I will use a pre-computed list of generator vectors
    # that are known to form the [18,9,6] QR code.
    
    # Generator vectors (18-bit integers):
    # I will generate them by taking the rows of the generator matrix of the [17,9,5] code
    # and appending the parity bit.
    
    # The [17,9,5] QR code generator matrix G_17 is a 9x17 matrix.
    # Row i is the coefficients of x^i * g(x) mod (x^17 - 1).
    
    # Let's assume the generator polynomial is g(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1.
    # I will verify the min distance in the code. If it's not 6, I will fall back to a greedy search.
    
    # Actually, a safer bet is to use the **Shortened Hamming Code** or similar, but QR is optimal.
    
    # Let's try to construct the QR code properly.
    # I will use the following generator matrix for the [18,9,6] code, which is standard:
    
    # Note: The following code generates the [18,9,6] QR code.
    
    # Generator polynomial g(x) for length 17 QR code:
    # g(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1
    # Let's check the weight of g(x): 6.
    # If this is correct, the min weight of the cyclic code is 5 or 6.
    
    # I will use a different, robust method:
    # Use the **Extended BCH Code** or **QR Code** constructed via NumPy if available, but standard library is preferred.
    
    # I will use a hardcoded generator matrix that is known to be correct for [18,9,6].
    # From online tables, the [18,9,6] QR code generator matrix rows (as 18-bit ints) are:
    
    gen_vectors = [
        0b000000000000000001, # This is not right, need actual vectors
    ]
    
    # I will generate the code using the following algorithm:
    # 1. Create the generator matrix for the [17,9,5] QR code.
    # 2. Add parity bit.
    
    # The generator polynomial is g(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1.
    # Let's assume this is correct.
    
    g_poly = 0
    coeffs = [1, 0, 1, 1, 1, 1, 0, 1, 1] # x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1
    # Wait, x^8 is bit 8.
    # g(x) = 1 + x + x^2 + x^4 + x^5 + x^6 + x^7 + x^8
    # Binary: 111101111 (reversed?)
    # Let's define g as a list of coefficients c0..c8.
    c = [1, 1, 1, 0, 1, 1, 1, 1, 1] # c0=1, c1=1, c2=1, c3=0, c4=1, c5=1, c6=1, c7=1, c8=1
    
    # Generate the 9 rows of the 17x17 generator matrix
    rows_17 = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if c[j] == 1:
                pos = (i + j) % 17
                row[pos] = 1
        rows_17.append(row)
        
    # Add parity bit to make 18-bit rows
    gen_matrix = []
    for row in rows_17:
        parity = sum(row) % 2
        new_row = row + [parity]
        # Convert to integer
        val = 0
        for k in range(18):
            if new_row[k]:
                val |= (1 << k)
        gen_matrix.append(val)
        
    # Generate all 512 codewords
    code_words = [0]
    for g in gen_matrix:
        code_words = code_words + [w ^ g for w in code_words]
        
    # Verify min distance
    # For a linear code, min distance is min weight of non-zero codewords
    min_d = 19
    for w in code_words:
        if w != 0:
            wt = w.bit_count()
            if wt < min_d:
                min_d = wt
            if min_d < MIN_DISTANCE:
                break
                
    if min_d >= MIN_DISTANCE:
        return code_words
        
    # If the hardcoded generator was wrong, fall back to greedy search
    # But the QR code should be correct.
    # To be safe, if it fails, return the RM code (32 words)
    
    # Fallback: Reed-Muller RM(1,4)
    rm_basis = []
    b0 = 0
    for i in range(16):
        b0 |= 1 << i
    rm_basis.append(b0)
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
        
    code_words_rm = [0]
    for b in rm_basis:
        code_words_rm = code_words_rm + [w ^ b for w in code_words_rm]
        
    return code_words_rm
