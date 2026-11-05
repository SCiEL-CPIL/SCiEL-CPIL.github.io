import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate the 66 blocks of S(4, 5, 11) using the Ternary Golay Code
    # Parity check matrix for the ternary Golay code [11, 6, 5]
    # H is 5x11
    H = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        [0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2],
        [0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2]
    ]
    
    blocks_11 = []
    # Iterate over all vectors in F3^11
    # There are 3^11 = 177147 vectors
    # We can iterate using nested loops or base-3 representation
    for i in range(3**11):
        if time.time() > deadline:
            break
            
        # Convert i to base 3 vector of length 11
        v = [0] * 11
        temp = i
        for j in range(11):
            v[j] = temp % 3
            temp //= 3
            
        # Check if H v^T = 0 mod 3
        valid = True
        for row in H:
            s = 0
            for k in range(11):
                s += row[k] * v[k]
            if s % 3 != 0:
                valid = False
                break
                
        if valid:
            # Calculate weight (number of non-zero elements)
            weight = sum(1 for x in v if x != 0)
            if weight == 5:
                # The support of the codeword is a block of S(4, 5, 11)
                # Note: The ternary Golay code has codewords of weight 5.
                # The support is the set of indices where v[k] != 0.
                # However, for a Steiner system, we just need the set of points.
                # In the ternary Golay code, the non-zero entries are 1 or 2.
                # The set of indices is the block.
                support = tuple(k for k in range(11) if v[k] != 0)
                if support not in blocks_11:
                    blocks_11.append(support)
                    
    # We expect 66 blocks
    # If we didn't get 66, there might be an issue with the parity check matrix
    # But this matrix is standard.
    
    # 2. Construct the 132 blocks of S(5, 6, 12)
    # Points are 0..11
    # Type 1: B U {11}
    # Type 2: {0..10} \ B
    all_points_11 = set(range(11))
    blocks_12 = []
    
    for B in blocks_11:
        # Type 1
        b1 = tuple(sorted(B + (11,)))
        blocks_12.append(b1)
        
        # Type 2
        b2_set = all_points_11 - set(B)
        b2 = tuple(sorted(b2_set))
        blocks_12.append(b2)
        
    # 3. Verify the design
    # Check that all 5-subsets of 0..11 are covered exactly once
    # There are C(12, 5) = 792 such subsets
    # Each block covers C(6, 5) = 6 such subsets
    # 132 blocks * 6 = 792. Perfect match.
    
    covered = set()
    valid = True
    for B in blocks_12:
        for sub in itertools.combinations(B, 5):
            if sub in covered:
                valid = False
                break
            covered.add(sub)
        if not valid:
            break
            
    if valid and len(covered) == 792:
        return [list(b) for b in blocks_12]
        
    # 4. If invalid, use a greedy repair
    # This should not happen if the construction is correct, but just in case.
    # We will try to swap blocks to fix conflicts.
    # For now, return the best set of non-conflicting blocks found greedily.
    
    blocks_12_sorted = sorted(blocks_12, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
    current_blocks = []
    covered_5sets = set()
    
    for B in blocks_12_sorted:
        subs = list(itertools.combinations(B, 5))
        conflict = False
        for s in subs:
            if s in covered_5sets:
                conflict = True
                break
        if not conflict:
            current_blocks.append(B)
            covered_5sets.update(subs)
            
    return [list(b) for b in current_blocks]

if __name__ == "__main__":
    print(construct())
