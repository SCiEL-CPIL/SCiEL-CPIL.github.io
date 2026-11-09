import random
import time
from itertools import combinations

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses randomized local search with bitset optimization.
    """
    # Target n
    target_n = 34
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead
    
    # Helper to check if a specific coloring is valid
    # This is slow, used only for final verification or small n
    def is_valid(n, colors):
        # colors is a list of 0/1 for triples in lex order
        # Reconstruct mask
        red_mask = [[0 for _ in range(n)] for _ in range(n)]
        blue_mask = [[0 for _ in range(n)] for _ in range(n)]
        
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if colors[idx] == 0: # Red
                        red_mask[a][b] |= (1 << c)
                        red_mask[b][a] |= (1 << c)
                        red_mask[a][c] |= (1 << b)
                        red_mask[c][a] |= (1 << b)
                        red_mask[b][c] |= (1 << a)
                        red_mask[c][b] |= (1 << a)
                    else: # Blue
                        blue_mask[a][b] |= (1 << c)
                        blue_mask[b][a] |= (1 << c)
                        blue_mask[a][c] |= (1 << b)
                        blue_mask[c][a] |= (1 << b)
                        blue_mask[b][c] |= (1 << a)
                        blue_mask[c][b] |= (1 << a)
                    idx += 1
        
        # Check Red K4
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Find d > c in intersection of red masks
                    candidates = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                    # Mask out bits <= c
                    candidates &= ~((1 << (c+1)) - 1)
                    if candidates:
                        return False
        
        # Check Blue K5
        # This is harder to optimize with simple bitsets for 5-cliques
        # We can use a recursive check or a simpler O(n^5) loop if n is small
        # For n=34, O(n^5) is too slow for frequent checks.
        # We will rely on the search to minimize violations.
        # For final check, we might need a more efficient Blue K5 check.
        # Or we can just trust the search if it runs long enough.
        
        # Efficient Blue K5 check:
        # For each 3-set {a,b,c}, find pairs {d,e} > c in BlueMask intersection
        # Then check the remaining 3 triples.
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Candidates for d,e must be in BlueMask[a][b], BlueMask[a][c], BlueMask[b][c]
                    # and > c
                    candidates = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    
                    # Get list of candidate vertices
                    cands = []
                    temp = candidates
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        cands.append(v)
                        temp ^= lsb
                    
                    # Check all pairs in cands
                    for i in range(len(cands)):
                        d = cands[i]
                        for j in range(i+1, len(cands)):
                            e = cands[j]
                            # Check remaining triples: (a,d,e), (b,d,e), (c,d,e)
                            if (blue_mask[a][d] & (1 << e)) and \
                               (blue_mask[b][d] & (1 << e)) and \
                               (blue_mask[c][d] & (1 << e)):
                                return False
        return True

    def find_violation(n, red_mask, blue_mask):
        # Find a Red K4
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    candidates = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    if candidates:
                        lsb = candidates & -candidates
                        d = lsb.bit_length() - 1
                        return ('red', (a, b, c, d))
        
        # Find a Blue K5
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    candidates = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    cands = []
                    temp = candidates
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        cands.append(v)
                        temp ^= lsb
                    for i in range(len(cands)):
                        d = cands[i]
                        for j in range(i+1, len(cands)):
                            e = cands[j]
                            if (blue_mask[a][d] & (1 << e)) and \
                               (blue_mask[b][d] & (1 << e)) and \
                               (blue_mask[c][d] & (1 << e)):
                                return ('blue', (a, b, c, d, e))
        return None

    def get_triple_index(n, a, b, c):
        # Lexicographic index of (a,b,c) in combinations(range(n), 3)
        # This is O(n) to compute, but we can precompute or use a formula
        # Formula for rank of combination (a,b,c) in (0..n-1, 3):
        # Rank = C(a, 3) + C(b, 2) + C(c, 1) ? No.
        # Standard rank:
        # r = 0
        # for i in range(a+1): r += C(n-1-i, 2) ... this is getting complicated.
        # Let's just precompute a dictionary for speed.
        return triple_indices[(a, b, c)]

    # Try different n from high to low
    for n in range(target_n, 9, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Random restart
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize random coloring

            # Correct initialization:
            num_triples = n * (n-1) * (n-2) // 6
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Build masks
            red_mask = [[0 for _ in range(n)] for _ in range(n)]
            blue_mask = [[0 for _ in range(n)] for _ in range(n)]
            triple_indices = {}
            
            idx = 0
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        triple_indices[(a, b, c)] = idx
                        if colors[idx] == 0:
                            red_mask[a][b] |= (1 << c)
                            red_mask[b][a] |= (1 << c)
                            red_mask[a][c] |= (1 << b)
                            red_mask[c][a] |= (1 << b)
                            red_mask[b][c] |= (1 << a)
                            red_mask[c][b] |= (1 << a)
                        else:
                            blue_mask[a][b] |= (1 << c)
                            blue_mask[b][a] |= (1 << c)
                            blue_mask[a][c] |= (1 << b)
                            blue_mask[c][a] |= (1 << b)
                            blue_mask[b][c] |= (1 << a)
                            blue_mask[c][b] |= (1 << a)
                        idx += 1
            
            # Local Search
            max_iters = 10000
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                violation = find_violation(n, red_mask, blue_mask)
                if violation is None:
                    # Found valid coloring
                    return {'n': n, 'colors': colors}
                
                color_type, vertices = violation
                # Pick a random triple from the violation
                if color_type == 'red':
                    a, b, c, d = vertices
                    # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
                    possible_triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                else:
                    a, b, c, d, e = vertices
                    # Triples: 10 of them
                    possible_triples = list(combinations((a,b,c,d,e), 3))
                
                t = random.choice(possible_triples)
                idx = triple_indices[t]
                
                # Flip color
                old_color = colors[idx]
                new_color = 1 - old_color
                colors[idx] = new_color
                
                # Update masks
                a, b, c = t
                if old_color == 0: # Was Red, now Blue
                    red_mask[a][b] &= ~(1 << c)
                    red_mask[b][a] &= ~(1 << c)
                    red_mask[a][c] &= ~(1 << b)
                    red_mask[c][a] &= ~(1 << b)
                    red_mask[b][c] &= ~(1 << a)
                    red_mask[c][b] &= ~(1 << a)
                    
                    blue_mask[a][b] |= (1 << c)
                    blue_mask[b][a] |= (1 << c)
                    blue_mask[a][c] |= (1 << b)
                    blue_mask[c][a] |= (1 << b)
                    blue_mask[b][c] |= (1 << a)
                    blue_mask[c][b] |= (1 << a)
                else: # Was Blue, now Red
                    blue_mask[a][b] &= ~(1 << c)
                    blue_mask[b][a] &= ~(1 << c)
                    blue_mask[a][c] &= ~(1 << b)
                    blue_mask[c][a] &= ~(1 << b)
                    blue_mask[b][c] &= ~(1 << a)
                    blue_mask[c][b] &= ~(1 << a)
                    
                    red_mask[a][b] |= (1 << c)
                    red_mask[b][a] |= (1 << c)
                    red_mask[a][c] |= (1 << b)
                    red_mask[c][a] |= (1 << b)
                    red_mask[b][c] |= (1 << a)
                    red_mask[c][b] |= (1 << a)

    # Fallback: Return a small valid coloring (e.g., n=9 from seed or smaller)
    # To ensure we don't return invalid, we use the seed logic for n=9
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-END
