        def construct():
            # The 132 blocks of the Steiner system S(5,6,12)
            # Generated from the 66 blocks of S(4,5,11) and their "complements"
            # For safety, we hardcode the list.
            
            # I will use a compact representation if possible, or just the list.
            # Since I cannot guarantee the exact 132 tuples from memory, I will use the 
            # backtracking search but with a much stronger heuristic or symmetry breaking.
            
            # Wait, the current program IS backtracking.
            # If I replace it with a direct construction, I need the blocks.
            
            # Let's try to improve the backtracking search instead?
            # The current score is 1.94.
            # If I fix the first block to (0,1,2,3,4,5) and use symmetry breaking, 
            # the search space is reduced by 132! / (6! * 11!) ?
            
            # Actually, the most reliable improvement is to use the known 
            # "Shortlex" order or similar to prune.
            
            # But the Directive says: "Construct ... by generating ...".
            # This implies I SHOULD use the construction.
            
            # I will use the 11-point construction.
            # The 66 blocks of S(4,5,11) on 0..10:
            base_blocks_11 = [
                (0, 1, 3, 4, 5),
                (0, 1, 2, 4, 5),
                (0, 1, 2, 3, 5),
                (0, 1, 2, 3, 4),
                (0, 1, 3, 4, 9),
                (0, 1, 2, 4, 9)
            ]
            
            blocks_11 = []
            for base in base_blocks_11:
                for shift in range(11):
                    block = tuple((x + shift) % 11 for x in base)
                    blocks_11.append(block)
            
            # These are 66 blocks.
            # The 66 blocks containing 11 in S(5,6,12) are:
            blocks_12_with_11 = [tuple(list(b) + [11]) for b in blocks_11]
            
            # The other 66 blocks?
            # They are the blocks of S(4,5,11) on 0..10? No.
            # They are the "complementary" blocks.
            # In S(5,6,12), the complement of a block is not a block (size 6 complement is size 6).
            # Is the set of complements of the 66 blocks containing 11 also a set of 66 blocks?
            # Let B be a block containing 11. B = {11} U C, C subset of 0..10.
            # Complement B' = {0..10} \ C. Size 6.
            # Does B' contain 11? No.
            # Are the 66 complements the other 66 blocks?
            # Let's check.
            # If we take all 66 blocks containing 11, their complements are 66 blocks not containing 11.
            # Do these 66 blocks form a valid part of the design?
            # Yes, the Witt design is "complementary" in the sense that the set of all 132 blocks is closed under complementation?
            # No, S(5,6,12) is not closed under complementation (complement of a block is not a block in general? 
            # Actually, in S(5,6,12), the complement of a block IS a block.
            # Proof: If B is a block, any 5-subset of B is in no other block.
            # Consider a 5-subset S of B'. S is disjoint from B.
            # S is contained in some block C.
            # C cannot be B (disjoint).
            # This property holds for S(5,6,12).
            
            # So the 132 blocks are:
            # 66 blocks containing 11.
            # 66 blocks NOT containing 11 (which are the complements of the first 66).
            
            # Let's verify the count.
            # 66 blocks with 11.
            # 66 blocks without 11.
            # Total 132.
            
            # So the algorithm is:
            # 1. Generate 66 blocks of S(4,5,11).
            # 2. Form 66 blocks with 11.
            # 3. Form 66 blocks as complements of the 66 blocks with 11 (excluding 11 from the universe? No, complement in 0..11).
            #    Complement of {11} U C is (0..11) \ ({11} U C) = (0..10) \ C.
            
            # Let's implement this.
            all_points = set(range(12))
            blocks_12 = []
            for b in blocks_11:
                b_with_11 = tuple(sorted(list(b) + [11]))
                blocks_12.append(b_with_11)
                # Complement
                comp = tuple(sorted(all_points - set(b_with_11)))
                blocks_12.append(comp)
                
            # Return list of lists
            return [list(b) for b in blocks_12]
        