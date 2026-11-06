import time
from collections import deque

def construct():
    """
    Returns the 132 blocks of the Steiner System S(5, 6, 12).
    Uses the orbit of a base block under the Mathieu group M12.
    """
    # Generators of M12
    # g1: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    g1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0)
    # g2: (0, 1, 2, 3, 4, 5)(6, 7, 8, 9, 10, 11)
    g2 = (1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6)
    # g3: (0, 2, 4, 6, 8, 10)(1, 3, 5, 7, 9, 11)
    g3 = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1)
    
    # Base block
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    # BFS to find all images of the base block
    seen_blocks = set()
    queue = deque([base_block])
    seen_blocks.add(base_block)
    
    while queue:
        block = queue.popleft()
        for g in [g1, g2, g3]:
            # Apply permutation g to block
            new_block = frozenset([g[x] for x in block])
            if new_block not in seen_blocks:
                seen_blocks.add(new_block)
                queue.append(new_block)
                
    # Convert to list of lists
    result = [list(b) for b in seen_blocks]
    
    return result

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
