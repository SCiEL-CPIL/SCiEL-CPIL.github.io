import time
import random
import numpy as np
from itertools import combinations

N = 16

# Precompute all edges and K4s
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

# Generate all K4s and their edge indices
k4s = []
k4_edges = []
for a, b, c, d in combinations(range(N), 4):
    k4s.append((a, b, c, d))
    e_indices = [
        edge_to_idx[(a, b)],
        edge_to_idx[(a, c)],
        edge_to_idx[(a, d)],
        edge_to_idx[(b, c)],
        edge_to_idx[(b, d)],
        edge_to_idx[(c, d)]
    ]
    k4_edges.append(e_indices)

# Map each edge to the list of K4 indices it participates in
edge_k4s = [[] for _ in range(len(edges))]
for k4_idx, e_indices in enumerate(k4_edges):
    for e_idx in e_indices:
        edge_k4s[e_idx].append(k4_idx)

# GF(16) construction
# GF(16) = GF(2)[x]/(x^4 + x + 1)
# Elements are 0-15, represented as 4-bit polynomials
# Multiplication in GF(16)
def gf16_mult(a, b):
    """Multiply two elements in GF(16) = GF(2)[x]/(x^4+x+1)"""
    result = 0
    a = a & 0xF
    b = b & 0xF
    for i in range(4):
        if b & (1 << i):
            result ^= a
        # Left shift a by 1 in GF(16)
        a = (a << 1) & 0xF
        if a & 0x10:  # If x^4 term appears, reduce by x^4 + x + 1
            a ^= 0x3  # x^4 = x + 1 in GF(16)
    return result & 0xF

# Precompute multiplication table
GF16_MULT_TABLE = [[gf16_mult(i, j) for j in range(16)] for i in range(16)]

# Trace function: Tr(x) = x + x^2 + x^4 + x^8 in GF(16)
# In GF(16), x^2 is squaring (Frobenius), x^4 = (x^2)^2, x^8 = (x^4)^2
def gf16_square(x):
    """Square in GF(16) - Frobenius automorphism"""
    # x^2 in GF(2)[x]/(x^4+x+1)
    # For x = a0 + a1*x + a2*x^2 + a3*x^3
    # x^2 = a0 + a1*x^2 + a2*x^4 + a3*x^6
    # x^4 = x + 1, x^6 = x^2 + x
    # So x^2 = a0 + a1*x^2 + a2*(x+1) + a3*(x^2+x)
    #        = (a0+a2) + (a2+a3)*x + (a1+a3)*x^2 + 0*x^3
    a0 = x & 1
    a1 = (x >> 1) & 1
    a2 = (x >> 2) & 1
    a3 = (x >> 3) & 1
    b0 = (a0 + a2) & 1
    b1 = (a2 + a3) & 1
    b2 = (a1 + a3) & 1
    b3 = 0
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)

def gf16_trace(x):
    """Trace from GF(16) to GF(2): Tr(x) = x + x^2 + x^4 + x^8"""
    x2 = gf16_square(x)
    x4 = gf16_square(x2)
    x8 = gf16_square(x4)
    # Sum in GF(2) = XOR
    return (x ^ x2 ^ x4 ^ x8) & 1

# Precompute trace values
GF16_TRACE = [gf16_trace(i) for i in range(16)]

# For the GF(16) coloring, we need to map to more colors
# Strategy: Use multiple "probes" - for edge {i,j}, compute trace(i*j), trace((i+1)*j), etc.
# and combine them to get a 4-bit (16-color) or higher value

def gf16_color(i, j, num_probes=4):
    """
    Color edge {i,j} using GF(16) structure.
    Use multiple shifted products to get more color variety.
    Returns an integer in 0..(2^num_probes - 1)
    """
    result = 0
    for k in range(num_probes):
        # Use different "shifts" to create independent probes
        a = (i + k * 5) % 16  # 5 is a generator-related value
        b = (j + k * 3) % 16  # 3 is coprime to 16
        prod = GF16_MULT_TABLE[a][b]
        trace_val = GF16_TRACE[prod]
        result |= (trace_val << k)
    return result

def gf16_color_v2(i, j):
    """
    Alternative GF(16) coloring: use the product directly mapped through a nonlinear function
    """
    prod = GF16_MULT_TABLE[i][j]
    # Map the 4-bit product to a color using a permutation
    # Use a fixed permutation of 0-15
    perm = [0, 5, 10, 15, 2, 7, 12, 1, 4, 9, 14, 3, 6, 11, 8, 13]
    return perm[prod]

def gf16_color_v3(i, j):
    """
    Use both i*j and (i+1)*(j+1) to create a 2-bit pair, then combine
    """
    p1 = GF16_MULT_TABLE[i][j]
    p2 = GF16_MULT_TABLE[(i+1)%16][(j+1)%16]
    # Combine two 4-bit values into one 8-bit value, then map to 0-15
    combined = (p1 << 4) | p2
    return combined % 16

def count_distinct_colors(colours, k4_idx):
    e_indices = k4_edges[k4_idx]
    c0 = colours[e_indices[0]]
    c1 = colours[e_indices[1]]
    c2 = colours[e_indices[2]]
    c3 = colours[e_indices[3]]
    c4 = colours[e_indices[4]]
    c5 = colours[e_indices[5]]
    return len({c0, c1, c2, c3, c4, c5})

def total_bad_count(colours):
    bad = 0
    for i in range(len(k4s)):
        if count_distinct_colors(colours, i) < 5:
            bad += 1
    return bad

def total_penalty(colours):
    pen = 0
    for i in range(len(k4s)):
        n_col = count_distinct_colors(colours, i)
        if n_col < 5:
            pen += (5 - n_col) ** 2
    return pen

def try_gf16_construction():
    """Try various GF(16) based constructions"""
    constructions = []
    
    # Construction 1: Basic trace-based with 4 probes (16 colors)
    def color1(i, j):
        return gf16_color(i, j, num_probes=4)
    constructions.append(color1)
    
    # Construction 2: Product with permutation
    def color2(i, j):
        return gf16_color_v2(i, j)
    constructions.append(color2)
    
    # Construction 3: Combined products
    def color3(i, j):
        return gf16_color_v3(i, j)
    constructions.append(color3)
    
    # Construction 4: Use i*j and i+j
    def color4(i, j):
        prod = GF16_MULT_TABLE[i][j]
        s = (i + j) % 16
        return (prod ^ s) % 16
    constructions.append(color4)
    
    # Construction 5: Use trace of i*j and trace of i+j
    def color5(i, j):
        t1 = GF16_TRACE[GF16_MULT_TABLE[i][j]]
        t2 = GF16_TRACE[(i + j) % 16]
        return t1 * 2 + t2
    constructions.append(color5)
    
    # Construction 6: Multiple independent probes with different shifts
    def color6(i, j):
        result = 0
        shifts = [(0, 0), (1, 2), (3, 5), (7, 11)]
        for k, (si, sj) in enumerate(shifts):
            a = (i + si) % 16
            b = (j + sj) % 16
            prod = GF16_MULT_TABLE[a][b]
            result |= (GF16_TRACE[prod] << k)
        return result
    constructions.append(color6)
    
    # Construction 7: Use the multiplicative group structure
    # Non-zero elements form GF(16)* which is cyclic of order 15
    # Map to Z_15 using a generator
    def gf16_log(x):
        if x == 0:
            return 0
        # Find discrete log base 3 (3 is a primitive element in GF(16))
        # 3^0=3, 3^1=9, 3^2=12, 3^3=6, 3^4=10, 3^5=15, 3^6=5, 3^7=1, 3^8=3... wait
        # Let's compute: 3^1=3, 3^2=3*3=9, 3^3=9*3=27=27 mod (x^4+x+1)
        # In GF(16): 3*3: 0011 * 0011
        # 3*3: result=0, i=0: 3&1=1, result^=3=3, a=6; i=1: 6&1=0, a=12; i=2: 12&1=0, a=24&0xF=8, 8&0x10=0; i=3: 8&1=0
        # So 3*3=6? Let me recompute
        # Actually let me just precompute the powers
        pass
    # Skip this for now, use simpler approach
    
    # Try each construction
    for color_func in constructions:
        colours = np.array([color_func(i, j) for i in range(N) for j in range(i+1, N)], dtype=np.int32)
        max_color = colours.max()
        if max_color < 16:  # Using at most 16 colors
            bad = total_bad_count(colours)
            if bad == 0:
                return colours.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for Python overhead
    
    # First, try GF(16) algebraic constructions
    gf16_result = try_gf16_construction()
    if gf16_result is not None:
        return gf16_result
    
    # If GF(16) didn't work, use simulated annealing
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        # Initialize random coloring
        colours = np.random.randint(0, K, size=120)
        bad_count = total_bad_count(colours)
        
        if bad_count == 0:
            return colours.tolist()
            
        # Local Search Parameters
        T = 50.0
        T_min = 0.1
        cooling = 0.99
        max_iterations = 200000
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            if restart > 0:
                if restart % 2 == 0:
                    # Random restart
                    colours = np.random.randint(0, K, size=120)
                    bad_count = total_bad_count(colours)
                    if bad_count == 0:
                        return colours.tolist()
                else:
                    # Perturb current solution
                    for _ in range(10):
                        e_idx = random.randint(0, 119)
                        colours[e_idx] = random.randint(0, K - 1)
                    bad_count = total_bad_count(colours)
                    if bad_count == 0:
                        return colours.tolist()
            
            T = 50.0
            # SA Loop
            while time.time() < deadline - 5:
                e_idx = random.randint(0, 119)
                old_c = colours[e_idx]
                new_c = random.randint(0, K - 1)
                
                if new_c == old_c:
                    continue
                
                affected_k4s = edge_k4s[e_idx]
                
                old_badness = 0
                for k4_idx in affected_k4s:
                    if count_distinct_colors(colours, k4_idx) < 5:
                        old_badness += 1
                        
                colours[e_idx] = new_c
                
                new_badness = 0
                for k4_idx in affected_k4s:
                    if count_distinct_colors(colours, k4_idx) < 5:
                        new_badness += 1
                        
                delta = new_badness - old_badness
                bad_count += delta
                
                if delta < 0:
                    pass
                elif delta == 0:
                    pass
                else:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        colours[e_idx] = old_c
                        bad_count -= delta
                        T *= cooling
                        if T < T_min:
                            T = T_min
                        continue
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                if bad_count == 0:
                    return colours.tolist()
                
    # Fallback to seed solution (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
