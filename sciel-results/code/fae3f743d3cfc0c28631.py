import time

def construct():
    # Generators for the Mathieu group M12
    # Permutations are represented as lists where p[i] is the image of i
    # g1: 0->1, 1->2, ..., 11->0
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    # g2: (0 1 3 7 11 5 2 6 10 4 8 9)
    g2 = [1, 3, 7, 11, 5, 2, 6, 10, 4, 8, 9, 0]
    
    def apply_perm(block, perm):
        # block is a tuple of points
        # perm is a list where perm[i] is the image of i
        # We want to map each point x in block to perm[x]
        return tuple(sorted(perm[x] for x in block))
    
    base_block = (0, 1, 2, 3, 4, 5)
    blocks_set = {base_block}
    blocks_list = [base_block]
    
    # Generate all blocks by applying generators
    changed = True
    while changed:
        changed = False
        current_size = len(blocks_list)
        for i in range(current_size):
            b = blocks_list[i]
            # Apply g1
            new_b1 = apply_perm(b, g1)
            if new_b1 not in blocks_set:
                blocks_set.add(new_b1)
                blocks_list.append(new_b1)
                changed = True
            # Apply g2
            new_b2 = apply_perm(b, g2)
            if new_b2 not in blocks_set:
                blocks_set.add(new_b2)
                blocks_list.append(new_b2)
                changed = True
                
    # The result should be 132 blocks
    return [list(b) for b in blocks_list]
