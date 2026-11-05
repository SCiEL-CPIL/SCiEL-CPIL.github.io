import time
import random
from itertools import combinations

def _generate_golay_generator_matrix():
    """
    Generates the 12x24 generator matrix for the extended binary Golay code.
    Uses the cyclic [23, 12, 7] Golay code defined by g(x) = 1 + x^3 + x^5 + x^6 + x^8 + x^9 + x^12.
    """
    # Coefficients of g(x) for x^0 to x^12
    # g(x) = 1 + x^3 + x^5 + x^6 + x^8 + x^9 + x^12
    g_poly = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    
    # Construct 12x23 generator matrix for the cyclic code
    G_23 = []
    for i in range(12):
        row = [0] * 23
        # Shift g_poly by i positions
        for j in range(13):
            if i + j < 23:
                row[i + j] = g_poly[j]
        G_23.append(row)
        
    # Extend to 24 bits by adding a parity bit to each row
    G_24 = []
    for row in G_23:
        parity = sum(row) % 2
        G_24.append(row + [parity])
        
    return G_24

def _witt_6_12_blocks():
    """
    Construct the Steiner system S(5,6,12) – the Witt design.
    Uses the algebraic connection to the binary Golay code.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate Golay Code Generator Matrix
    G = _generate_golay_generator_matrix()
    
    # 2. Generate all codewords and collect octads (weight 8) and dodecads (weight 12)
    octads = []
    dodecads = []
    
    # There are 2^12 = 4096 codewords
    for mask in range(4096):
        # Compute codeword c = mask * G mod 2
        c = [0] * 24
        for i in range(12):
            if (mask >> i) & 1:
                # Add row i of G to c
                row = G[i]
                for j in range(24):
                    c[j] ^= row[j]
        
        # Calculate weight
        weight = sum(c)
        
        if weight == 8:
            # Store support as a tuple of indices
            support = tuple(j for j in range(24) if c[j] == 1)
            octads.append(support)
        elif weight == 12:
            support = tuple(j for j in range(24) if c[j] == 1)
            dodecads.append(support)
            
        if time.time() > deadline:
            # Fallback if generation takes too long (unlikely)
            break

    if not dodecads:
        # Fallback to greedy if something went wrong
        return _fallback_greedy()

    # 3. Pick a dodecad to define the 12 points
    D = dodecads[0]
    points = set(D)
    
    # 4. Generate blocks
    # The blocks of S(5,6,12) on the points of D are the intersections of D with octads
    # that have intersection size 6.
    blocks_set = set()
    
    for O in octads:
        O_set = set(O)
        intersection = points.intersection(O_set)
        if len(intersection) == 6:
            # Store as sorted tuple for hashing
            block = tuple(sorted(intersection))
            blocks_set.add(block)
            
        if time.time() > deadline:
            break

    # Convert to list of lists
    blocks = [list(b) for b in blocks_set]
    
    # Verify we have 132 blocks
    if len(blocks) == 132:
        return blocks
    else:
        # If the algebraic construction failed for some reason, fallback
        return _fallback_greedy()

def _fallback_greedy():
    """
    Fallback greedy search if algebraic construction fails.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    all_6subsets = list(combinations(range(12), 6))
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for sub in combinations(b, 5):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    def greedy_packing(order):
        covered = 0
        blocks = []
        for idx in order:
            mask = block_masks[idx]
            if (covered & mask) == 0:
                blocks.append(all_6subsets[idx])
                covered |= mask
                if len(blocks) == 132:
                    break
        return blocks

    rng = random.Random(42)
    num_blocks = len(all_6subsets)
    best_blocks = []
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        order = list(range(num_blocks))
        rng.shuffle(order)
        blocks = greedy_packing(order)
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
            if len(best_blocks) == 132:
                break
                
    return [list(b) for b in best_blocks]

def construct():
    """
    Construct a packing of 6-subsets on 12 points with no repeated 5-subset.
    Returns a list of distinct 6-subsets of 0..11.
    """
    return _witt_6_12_blocks()
