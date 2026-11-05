import time
import itertools
import numpy as np
import random

def _generate_witt_blocks():
    """
    Generates the 132 blocks of the Steiner System S(5, 6, 12)
    using the action of the Mathieu group M12 on a base block.
    """
    # Permutations for M12 generators
    # g1: (0 1 2 3 4 5 6 7 8 9 10 11)
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    # g2: (1 2 4 8 10)(3 6 11 7 5)
    # Let's define g2 carefully:
    # 0->0, 1->2, 2->4, 3->6, 4->8, 5->7, 6->11, 7->5, 8->10, 9->9, 10->1, 11->3
    # Wait, 9 is fixed?
    # Let's use a known pair of generators.
    # From GAP:
    # g1 = (1,2,3,4,5,6,7,8,9,10,11,12)
    # g2 = (1,2,3,4,5,6,7,8,9,10,11,12)
    
    # I will use a safer set of generators if I can find them.
    # Otherwise, I will use the hardcoded list.
    
    # Let's try the BFS with these two and check the count.
    # If count != 132, return None.
    
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    blocks = set()
    queue = [base_block]
    blocks.add(base_block)
    
    # Generators
    # g1: 12-cycle
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    # g2: 5-cycle product
    # Let's use (1 2 3 4 5) in 1-based, so (0 1 2 3 4) in 0-based?
    # M12 is sharply 4-transitive.
    
    # I will use the following generators which are known to generate M12:
    # p1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # p2 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    
    # Actually, let's just hardcode the 132 blocks to be safe.
    # This is the most robust O(1) solution.
    # I will generate them using a known valid list.
    
    # Since I cannot be 100% sure of the generators from memory,
    # I will use the Greedy Fallback but with a much better heuristic:
    # "Maximal Independent Set" in the conflict graph, using a specific ordering.
    
    # However, the prompt inspiration strongly suggests the algebraic construction.
    # I will implement the BFS with the generators and check the result.
    
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    g2 = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11] # (1 2 3 4 5) (6 7 8 9 10) (0) (11)
    
    while queue:
        b = queue.pop()
        b1 = frozenset([g1[i] for i in b])
        if b1 not in blocks:
            blocks.add(b1)
            queue.append(b1)
        b2 = frozenset([g2[i] for i in b])
        if b2 not in blocks:
            blocks.add(b2)
            queue.append(b2)
            
    if len(blocks) == 132:
        return [list(b) for b in blocks]
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Try direct construction first
    witt_blocks = _generate_witt_blocks()
    if witt_blocks is not None:
        # Verify validity
        all_5_subsets = set(itertools.combinations(range(12), 5))
        seen_5_subsets = set()
        valid = True
        for b in witt_blocks:
            for s in itertools.combinations(b, 5):
                if s in seen_5_subsets:
                    valid = False
                    break
                seen_5_subsets.add(s)
            if not valid:
                break
        
        if valid and len(witt_blocks) == 132:
            return witt_blocks

    # Fallback to original logic if construction fails
    points = range(12)
    all_blocks = list(itertools.combinations(points, 6))
    all_5_subsets = list(itertools.combinations(points, 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    block_5_indices = []
    for b in all_blocks:
        indices = [subset_to_idx[s] for s in itertools.combinations(b, 5)]
        block_5_indices.append(indices)
        
    # Greedy fallback
    best_blocks = []
    covered = np.zeros(len(all_5_subsets), dtype=bool)
    blocks = []
    for j, indices in enumerate(block_5_indices):
        if not np.any(covered[indices]):
            blocks.append(list(all_blocks[j]))
            covered[indices] = True
    if len(blocks) > len(best_blocks):
        best_blocks = blocks
        
    return best_blocks
