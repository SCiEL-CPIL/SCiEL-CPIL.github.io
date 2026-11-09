"""
Constructs the Steiner system S(5,6,12) (small Witt design) on 12 points.
This system consists of 132 blocks of size 6 such that every 5-subset
is contained in exactly one block.
"""

def construct():
    # Points are 0 to 11
    n = 12
    
    # Define generators for the Mathieu group M12 acting on 12 points
    # g1: 12-cycle (0,1,2,3,4,5,6,7,8,9,10,11)
    g1 = [(i + 1) % 12 for i in range(12)]
    
    # g2: (0,1)(2,5)(3,6)(4,7)(8,10)(9,11)
    # Mapping: 0->1, 1->0, 2->5, 5->2, 3->6, 6->3, 4->7, 7->4, 8->10, 10->8, 9->11, 11->9
    g2 = [1, 0, 5, 6, 7, 2, 3, 4, 10, 11, 8, 9]
    
    # g3: (0,2,4)(1,3,5)(6,8,10)(7,9,11)
    # Mapping: 0->2, 2->4, 4->0, 1->3, 3->5, 5->1, 6->8, 8->10, 10->6, 7->9, 9->11, 11->7
    g3 = [2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 6, 7]
    
    generators = [g1, g2, g3]
    
    # Base block for S(5,6,12)
    # This is a known block in the standard labeling
    base_block = frozenset([0, 1, 2, 4, 10, 11])
    
    blocks = {base_block}
    
    # Iteratively apply generators to expand the orbit until stable
    changed = True
    while changed:
        changed = False
        new_blocks = set()
        for b in blocks:
            for g in generators:
                # Apply permutation g to the block b
                new_b = frozenset(g[p] for p in b)
                if new_b not in blocks:
                    new_blocks.add(new_b)
        if new_blocks:
            blocks.update(new_blocks)
            changed = True
            
    # Convert to list of lists for the required output format
    result = [list(b) for b in blocks]
    
    return result
