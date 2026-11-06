import itertools
import time

def construct():
    """
    Constructs the Steiner System S(5,6,12) (Witt Design) on 12 points.
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    """
    
    # Step 1: Construct S(4,5,11) using 6 base blocks under cyclic group Z_11
    # The 6 base blocks are chosen such that their orbits under addition mod 11
    # partition the set of all 4-subsets of {0..10} into 66 5-subsets (blocks).
    
    base_blocks = [
        (0, 1, 2, 4, 10),
        (0, 1, 3, 4, 8),
        (0, 1, 3, 5, 9),
        (0, 2, 3, 5, 10),
        (0, 2, 4, 5, 8),
        (0, 2, 4, 7, 9)
    ]
    
    s4511_blocks = []
    for bb in base_blocks:
        for shift in range(11):
            block = tuple(sorted((x + shift) % 11 for x in bb))
            s4511_blocks.append(block)
            
    # Step 2: Construct S(5,6,12) from S(4,5,11)
    # Points 0..10 in S(4,5,11), plus point 11.
    # For each block B in S(4,5,11):
    # 1. B + {11}
    # 2. (0..10 \ B)
    
    w12_blocks = []
    all_points_11 = set(range(11))
    
    for b in s4511_blocks:
        # b is a tuple of 5 elements from 0..10
        # Block 1: b plus 11
        w12_blocks.append(tuple(sorted(list(b) + [11])))
        
        # Block 2: complement of b in 0..10
        comp = tuple(sorted(all_points_11 - set(b)))
        w12_blocks.append(comp)
    
    # The result should be 132 blocks
    # Convert to list of lists as per interface
    return [list(b) for b in w12_blocks]

if __name__ == "__main__":
    # Quick self-test
    blocks = construct()
    print(f"Constructed {len(blocks)} blocks")
    # Verify validity
    if len(blocks) == 132:
        seen_5sets = set()
        valid = True
        for b in blocks:
            if len(set(b)) != 6:
                valid = False
                break
            for s in itertools.combinations(b, 5):
                if s in seen_5sets:
                    valid = False
                    break
                seen_5sets.add(s)
            if not valid:
                break
        print(f"Valid: {valid}, Unique 5-sets: {len(seen_5sets)}")
    else:
        print("Invalid count")
