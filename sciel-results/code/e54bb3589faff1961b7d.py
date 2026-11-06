"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Points are 0..10 and 11 (representing infinity)
# We will use the group PSL(2,11) acting on the projective line over GF(11)
# Points: 0, 1, ..., 10, inf (mapped to 11)

def _psl2_11_elements():
    """Generate all elements of PSL(2,11)."""
    elements = set()
    # Matrices [[a, b], [c, d]] with ad - bc = 1 mod 11
    # We can generate them by iterating a, b, c, d and checking determinant.
    # To ensure unique representatives in PSL, we usually identify M and -M.
    # But for permutation generation, we just need the distinct permutations.
    # The group order is 660.
    
    def apply_perm(p, x):
        # p is (a, b, c, d)
        a, b, c, d = p
        if x == 11: # infinity
            # (a*inf + b) / (c*inf + d) = a/c
            if c == 0:
                return 11 # a/0 = inf
            else:
                return (a * pow(c, 9, 11)) % 11
        else:
            # (a*x + b) / (c*x + d)
            num = (a * x + b) % 11
            den = (c * x + d) % 11
            if den == 0:
                return 11
            else:
                return (num * pow(den, 9, 11)) % 11

    perms = set()
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    if (a * d - b * c) % 11 != 1:
                        continue
                    # Check if this matrix is non-identity and not duplicate
                    # We just collect the permutations
                    perm = tuple(apply_perm((a, b, c, d), x) for x in range(12))
                    perms.add(perm)
    return list(perms)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate PSL(2,11) permutations
    # This might take a second or two.
    perms = _psl2_11_elements()
    
    # 2. Try to find a base block whose orbit under PSL(2,11) forms S(5,6,12)
    # The number of blocks in S(5,6,12) is 132.
    # |PSL(2,11)| = 660.
    # If the stabilizer of a block has order 5, orbit size is 132.
    
    # Candidate base blocks. We need to test a few.
    # Known base blocks for Witt design S(5,6,12) often involve small numbers.
    # Let's try a few candidates.
    candidates = [
        (0, 1, 3, 4, 5, 9),
        (0, 1, 2, 4, 5, 8),
        (0, 1, 3, 4, 6, 10),
        (0, 1, 2, 3, 5, 8),
        (0, 1, 2, 4, 6, 9),
        (0, 1, 3, 5, 6, 10),
        (0, 2, 3, 4, 5, 9),
        (0, 1, 3, 4, 5, 10),
    ]
    
    # Precompute 5-subsets for validation
    # We don't need to precompute all 792, just check the generated set.
    
    for base in candidates:
        # Generate orbit
        orbit = set()
        for p in perms:
            # Apply permutation to base block
            new_block = tuple(sorted(p[base[i]] for i in range(6)))
            orbit.add(new_block)
        
        # Check if orbit size is 132
        if len(orbit) != 132:
            continue
            
        # Check if it's a valid S(5,6,12)
        # Every 5-subset must appear exactly once.
        # There are C(12,5) = 792 5-subsets.
        # Each block contains 6 5-subsets. 132 * 6 = 792.
        # So if we can cover all 792 without repetition, it's valid.
        
        covered = bytearray(792)
        # Map 5-subsets to indices for fast checking
        # We can generate the 5-subsets of each block and check coverage.
        # To be fast, we can just check for collisions.
        # If no collision and count is 132, it's valid?
        # Yes, because total capacity is exactly 792.
        
        valid = True
        seen_fives = set()
        for block in orbit:
            for i in range(6):
                # Get 5-subset by removing element i
                sub = block[:i] + block[i+1:]
                if sub in seen_fives:
                    valid = False
                    break
                seen_fives.add(sub)
            if not valid:
                break
        
        if valid and len(seen_fives) == 792:
            # Success!
            return [list(b) for b in orbit]
            
    # Fallback: Greedy search if algebraic construction fails
    # This should not happen if a candidate is correct.
    
    FIVE_SUBSETS = list(combinations(range(12), 5))
    FIVE_TO_IDX = {s: i for i, s in enumerate(FIVE_SUBSETS)}
    SIX_SUBSETS = list(combinations(range(12), 6))
    SIX_TO_FIVE_IDX = []
    for b in SIX_SUBSETS:
        idxs = [FIVE_TO_IDX[s] for s in combinations(b, 5)]
        SIX_TO_FIVE_IDX.append(idxs)
        
    covered = bytearray(792)
    current_blocks = []
    
    for idx in range(len(SIX_SUBSETS)):
        idxs = SIX_TO_FIVE_IDX[idx]
        if not any(covered[i] for i in idxs):
            current_blocks.append(list(SIX_SUBSETS[idx]))
            for i in idxs:
                covered[i] = 1
                
    return current_blocks
# EVOLVE-BLOCK-END
