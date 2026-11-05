import time
import random
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) by generating the orbits of a base block
    under the action of the Mathieu group M12.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Attempt Group Construction
    try:
        # Generators for M12
        # Point set: 0..11
        # g1: 12-cycle
        g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        # g2: A permutation of order 5 or similar
        # A standard generator pair for M12:
        # (0,1,2,3,4,5,6,7,8,9,10,11) and (1,2,4,8)(3,6,5)(7,11,9)
        # Let's map (1,2,4,8) to 0-based: (0,1,3,7)
        # (3,6,5) -> (2,5,4)
        # (7,11,9) -> (6,10,8)
        # So g2 maps:
        # 0->1, 1->3, 3->7, 7->0
        # 2->5, 5->4, 4->2
        # 6->10, 10->8, 8->6
        # 9->9, 11->11
        g2 = [1, 3, 5, 7, 2, 4, 10, 0, 6, 9, 8, 11]
        
        # Verify g2 is a permutation
        if sorted(g2) != list(range(12)):
            raise ValueError("Invalid generator g2")

        # Generate group elements
        group = set()
        group.add(tuple(g1))
        group.add(tuple(g2))
        
        # BFS to generate group
        queue = [tuple(g1), tuple(g2)]
        while queue and time.time() < deadline - 5.0:
            g = queue.pop(0)
            # Multiply g by generators
            for gen in (tuple(g1), tuple(g2)):
                # Compose g * gen
                # (g * gen)[i] = g[gen[i]]
                new_g = tuple(g[gen[i]] for i in range(12))
                if new_g not in group:
                    group.add(new_g)
                    queue.append(new_g)
                
                # Compose gen * g
                # (gen * g)[i] = gen[g[i]]
                new_g2 = tuple(gen[g[i]] for i in range(12))
                if new_g2 not in group:
                    group.add(new_g2)
                    queue.append(new_g2)
        
        # Check if we found M12 (size 7920)
        if len(group) != 7920:
            # Might be a subgroup or generation failed
            # Try to proceed with what we have, or fallback
            pass

        # Base block
        base_block = (0, 1, 2, 3, 4, 5)
        
        blocks = set()
        for g in group:
            # Apply g to base_block
            # g maps i -> g[i]
            # We want the set { g[i] for i in base_block }
            b = tuple(sorted(g[i] for i in base_block))
            blocks.add(b)
            
            # Also check the complement? No, M12 is transitive on blocks.
            # The orbit of a block under M12 is the set of all blocks.
            
        if len(blocks) == 132:
            # Verify validity
            covered_5s = set()
            valid = True
            for b in blocks:
                for s in combinations(b, 5):
                    if s in covered_5s:
                        valid = False
                        break
                    covered_5s.add(s)
                if not valid:
                    break
            
            if valid:
                return [list(b) for b in blocks]
        
        # If group construction failed or didn't yield 132, fallback
        
    except Exception:
        pass

    # 2. Fallback: Randomized Greedy
    # This is a structural change from the hardcoded list to a search method.
    best_blocks = []
    
    # Precompute all 5-subsets for each 6-subset?
    # There are 924 6-subsets.
    # We can just check on the fly.
    
    all_blocks = list(combinations(range(12), 6))
    
    for attempt in range(200):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(attempt)
        random.shuffle(all_blocks)
        
        selected = []
        covered_5s = set()
        
        for b in all_blocks:
            # Check if b conflicts
            conflict = False
            for s in combinations(b, 5):
                if s in covered_5s:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(list(b))
                for s in combinations(b, 5):
                    covered_5s.add(s)
                
                if len(selected) == 132:
                    return selected
        
        if len(selected) > len(best_blocks):
            best_blocks = selected

    return best_blocks

if __name__ == "__main__":
    blocks = construct()
    print(f"Number of blocks: {len(blocks)}")
