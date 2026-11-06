import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Attempt to construct the Steiner System S(5, 6, 12) directly
    # using known base blocks for S(4, 5, 11).
    
    # Base blocks for S(4, 5, 11) on points 0..10
    # These are developed mod 11 to get 66 blocks.
    base_blocks = [
        (0, 1, 3, 7, 9),
        (0, 2, 3, 5, 8),
        (0, 1, 4, 6, 10),
        (0, 2, 4, 7, 10),
        (0, 1, 2, 5, 6),
        (0, 1, 5, 8, 10)
    ]
    
    # Generate all 66 blocks of S(4, 5, 11)
    s4511_blocks = []
    for base in base_blocks:
        for i in range(11):
            block = tuple(sorted((x + i) % 11 for x in base))
            s4511_blocks.append(block)
            
    # Remove duplicates just in case (should be unique)
    s4511_blocks = list(set(s4511_blocks))
    
    if len(s4511_blocks) != 66:
        # Fallback to greedy if generation failed
        return greedy_construct()

    # Construct S(5, 6, 12) blocks
    # Points are 0..11, where 11 is the point at infinity.
    # Type 1: {11} U B for B in S(4, 5, 11)
    # Type 2: {0..10} \ B for B in S(4, 5, 11)
    
    blocks = []
    points_0_10 = set(range(11))
    
    for B in s4511_blocks:
        # Type 1
        b1 = tuple(sorted(list(B) + [11]))
        blocks.append(b1)
        
        # Type 2
        b2_set = points_0_10 - set(B)
        b2 = tuple(sorted(b2_set))
        blocks.append(b2)
        
    # Verify the constructed design
    if len(blocks) != 132:
        return greedy_construct()
        
    # Check for duplicate blocks
    if len(set(blocks)) != 132:
        return greedy_construct()
        
    # Check that no 5-subset is repeated
    covered_5s = set()
    for b in blocks:
        for s in combinations(b, 5):
            if s in covered_5s:
                # Invalid design, fallback
                return greedy_construct()
            covered_5s.add(s)
            
    # If we get here, the design is valid and optimal
    return [list(b) for b in blocks]

def greedy_construct():
    """
    Fallback greedy construction if the direct construction fails.
    """
    points = range(12)
    all_6_subsets = list(combinations(points, 6))
    covered = set()
    blocks = []
    
    for b in all_6_subsets:
        block_5s = list(combinations(b, 5))
        if not any(s in covered for s in block_5s):
            blocks.append(b)
            covered.update(block_5s)
            
    return [list(b) for b in blocks]
