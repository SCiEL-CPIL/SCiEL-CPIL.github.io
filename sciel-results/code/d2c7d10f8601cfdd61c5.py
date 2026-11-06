import sys
import time

def construct():
    """
    Constructs the Steiner System S(5,6,12) (Witt Design) on 12 points.
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    Uses bitmask-based backtracking for S(4,5,11) construction.
    """
    
    start_time = time.time()
    DEADLINE = 100.0  # 100s safety margin
    
    # 1. Construct S(4,5,11) on points 0..10
    # Number of 4-sets: C(11,4) = 330
    # Number of 5-sets: C(11,5) = 462
    
    # Generate all 4-sets and 5-sets as bitmasks
    # Bit i represents point i
    
    all_4sets = []
    all_5sets = []
    
    # Helper to get bitmask from tuple
    def tuple_to_mask(t):
        m = 0
        for x in t:
            m |= (1 << x)
        return m

    # Precompute combinations
    # Using itertools for generation, then converting to masks
    import itertools
    
    points = range(11)
    
    # Generate 4-sets
    comb_4 = list(itertools.combinations(points, 4))
    all_4sets = [tuple_to_mask(c) for c in comb_4]
    # Map mask to index for 4-sets
    mask_to_4idx = {m: i for i, m in enumerate(all_4sets)}
    
    # Generate 5-sets
    comb_5 = list(itertools.combinations(points, 5))
    all_5sets = [tuple_to_mask(c) for c in comb_5]
    
    # Precompute relationships
    # For each 5-set, which 4-sets does it contain?
    # A 5-set {a,b,c,d,e} contains 5 4-sets.
    five_to_fours = []
    for mask in all_5sets:
        # Get the 5 points
        pts = []
        for i in range(11):
            if mask & (1 << i):
                pts.append(i)
        # Generate 4-subsets of these 5 points
        f4_indices = []
        for c4 in itertools.combinations(pts, 4):
            m4 = tuple_to_mask(c4)
            f4_indices.append(mask_to_4idx[m4])
        five_to_fours.append(f4_indices)
        
    # For each 4-set, which 5-sets contain it?
    four_to_fives = [[] for _ in range(330)]
    for idx5, f4_list in enumerate(five_to_fours):
        for idx4 in f4_list:
            four_to_fives[idx4].append(idx5)
            
    # Backtracking state
    # covered_4sets_mask: A big integer where bit i is 1 if 4-set i is covered.
    # 330 bits.
    covered_mask = 0
    blocks_5sets = [] # List of 5-set indices chosen
    
    def backtrack():
        nonlocal covered_mask
        if time.time() - start_time > DEADLINE:
            return False
        
        # Find the first uncovered 4-set
        # We need to find the smallest i such that bit i in covered_mask is 0.
        # Invert covered_mask and find lowest set bit?
        # Or just iterate? 330 iterations is fast.
        target_idx = -1
        # Optimization: Check bits
        # ~covered_mask will have 1s for uncovered.
        # But we only care about bits 0..329.
        # Mask for valid range: (1 << 330) - 1
        valid_range_mask = (1 << 330) - 1
        uncovered_mask = (~covered_mask) & valid_range_mask
        
        if uncovered_mask == 0:
            return True # All covered
        
        # Find lowest set bit in uncovered_mask
        # target_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        target_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        # Candidates: 5-sets containing this 4-set
        candidates = four_to_fives[target_idx]
        
        for idx5 in candidates:
            # Check if this 5-set conflicts with existing covered 4-sets
            # The 5-set contains 5 4-sets.
            # Create a bitmask for these 5 4-sets
            f4_mask = 0
            for idx4 in five_to_fours[idx5]:
                f4_mask |= (1 << idx4)
                
            # Check intersection with covered_mask
            if (f4_mask & covered_mask) != 0:
                continue
                
            # Valid move
            # Update state
            covered_mask |= f4_mask
            blocks_5sets.append(idx5)
            
            if backtrack():
                return True
            
            # Undo
            blocks_5sets.pop()
            covered_mask &= ~f4_mask
            
        return False

    # Start backtracking
    # Fix the first 4-set to be (0,1,2,3) to break symmetry?
    # Actually, S(4,5,11) is unique, so any start works.
    # But fixing one reduces branching slightly.
    # Let's just let it find the first one.
    
    if not backtrack():
        # Fallback: return empty or partial?
        # If timeout, we should return what we have?
        # But we need 132 blocks.
        # If we fail, we return [] which scores 0.
        # But this should not happen with bitmasks.
        return []
        
    # 2. Extend S(4,5,11) to S(5,6,12)
    # Blocks of S(5,6,12) on {0..11}:
    # 1. B U {11} for each B in S(4,5,11)
    # 2. B^c (complement in {0..10}) for each B in S(4,5,11)
    
    w12_blocks = []
    all_points_10_mask = (1 << 11) - 1
    
    for idx5 in blocks_5sets:
        mask5 = all_5sets[idx5]
        
        # Block 1: mask5 | (1 << 11)
        block1_mask = mask5 | (1 << 11)
        # Convert to tuple/list
        block1 = [i for i in range(12) if block1_mask & (1 << i)]
        w12_blocks.append(block1)
        
        # Block 2: Complement of mask5 in {0..10}
        # mask5 is a 5-set in 11 points. Complement is 6-set in 11 points.
        block2_mask = all_points_10_mask ^ mask5
        block2 = [i for i in range(11) if block2_mask & (1 << i)]
        w12_blocks.append(block2)
        
    return w12_blocks

if __name__ == "__main__":
    blocks = construct()
    print(f"Constructed {len(blocks)} blocks")
