"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Weight-2 construction: all vectors with exactly two non-zero entries, each in {+1, -1}
    # These have norm sqrt(2), and pairwise distance is at least sqrt(2) for distinct vectors
    # (since any two distinct weight-2 vectors differ in at least one position, 
    #  and the squared distance is sum of squared differences which is >= 2)
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # This gives C(11,2) * 4 = 55 * 4 = 220 vectors
    
    # We can also add weight-4 vectors if time permits, but weight-2 alone gives 220
    # which is a solid baseline. Let's check if we can add more.
    
    # For weight-4 vectors, we need supports of size 4 that intersect in at most 2 elements
    # with all other selected supports. This is more complex.
    
    # Let's try to add some weight-4 vectors using a greedy approach
    all_supports_w4 = list(itertools.combinations(range(D), 4))
    selected_supports_w4 = []
    
    for s in all_supports_w4:
        valid = True
        for t in selected_supports_w4:
            intersection = len(set(s).intersection(t))
            if intersection > 2:
                valid = False
                break
        if valid:
            selected_supports_w4.append(s)
            if len(selected_supports_w4) >= 32:
                break
    
    # Add weight-4 vectors from selected supports
    for s in selected_supports_w4:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    # Check time and return what we have
    if time.time() > deadline - 10:
        result = np.array(vectors, dtype=np.int64)
        return result
    
    # Verify the configuration: min pairwise distance >= max norm
    # All vectors have norm sqrt(2) or sqrt(4)=2
    # Max norm is 2 (from weight-4 vectors)
    # Min pairwise distance: for weight-2 vectors, min dist is sqrt(2)
    # But we need min dist >= max norm = 2
    # So weight-2 vectors alone don't work if we have weight-4 vectors!
    
    # Let me reconsider: the condition is min pairwise distance >= max norm
    # If we have weight-4 vectors (norm 2), then min distance must be >= 2
    # Two weight-2 vectors can have distance sqrt(2) < 2, so they violate the condition
    
    # So we should either:
    # 1. Use only weight-2 vectors (norm sqrt(2), min dist sqrt(2) >= sqrt(2) ✓) -> 220 vectors
    # 2. Use only weight-4 vectors (norm 2, need min dist >= 2) -> fewer vectors
    # 3. Mix carefully
    
    # Option 1 is simpler and gives 220 vectors. Let's go with that.
    # Remove weight-4 vectors to ensure validity
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    # Verify: all norms are sqrt(2), min distance between distinct vectors
    # Two distinct weight-2 vectors: they differ in at least one position
    # If they share no positions: distance squared = 2+2=4, dist=2
    # If they share one position with same sign: distance squared = 0+2=2, dist=sqrt(2)
    # If they share one position with different sign: distance squared = 4+2=6, dist=sqrt(6)
    # If they share both positions with same signs: same vector, not distinct
    # If they share both positions with one sign different: distance squared = 2+2=4, dist=2
    
    # So min distance is sqrt(2), max norm is sqrt(2). Condition holds: sqrt(2) >= sqrt(2) ✓
    
    return result

# EVOLVE-BLOCK-END
