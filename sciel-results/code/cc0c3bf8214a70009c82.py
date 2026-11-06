"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    
    deadline = time.time() + 140
    
    def make_triple_index(n):
        """Map (a,b,c) -> index in lexicographic order."""
        triples = list(combinations(range(n), 3))
        idx = {}
        for i, t in enumerate(triples):
            idx[t] = i
        return triples, idx
    
    def check_valid(n, colors):
        """Check no red K4 and no blue K5. Returns True if valid."""
        triples, tidx = make_triple_index(n)
        
        # Check red K4
        for quad in combinations(range(n), 4):
            count_red = 0
            for tri in combinations(quad, 3):
                if colors[tidx[tri]] == 0:
                    count_red += 1
            if count_red == 4:
                return False
        
        # Check blue K5
        for quint in combinations(range(n), 5):
            count_blue = 0
            for tri in combinations(quint, 3):
                if colors[tidx[tri]] == 1:
                    count_blue += 1
            if count_blue == 10:
                return False
        
        return True
    
    def try_modular(n, m, threshold):
        """Color triple (a,b,c) red if (a+b+c)%m < threshold, else blue."""
        triples, _ = make_triple_index(n)
        colors = [0 if (a + b + c) % m < threshold else 1 for a, b, c in triples]
        return colors
    
    def try_xor(n, bits):
        """Color based on XOR of vertex labels."""
        triples, _ = make_triple_index(n)
        colors = []
        for a, b, c in triples:
            x = a ^ b ^ c
            colors.append(0 if x < (1 << bits) else 1)
        return colors
    
    # Try structured colorings for decreasing n
    for n in range(34, 10, -1):
        if time.time() > deadline:
            break
        
        # Try modular colorings
        for m in range(2, min(60, n + 10)):
            if time.time() > deadline:
                break
            for threshold in range(1, m):
                colors = try_modular(n, m, threshold)
                if check_valid(n, colors):
                    return {'n': n, 'colors': colors}
        
        # Try XOR-based colorings
        for bits in range(1, 8):
            if time.time() > deadline:
                break
            colors = try_xor(n, bits)
            if check_valid(n, colors):
                return {'n': n, 'colors': colors}
    
    # Fallback to small valid coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
