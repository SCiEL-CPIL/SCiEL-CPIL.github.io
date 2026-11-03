<<<<<<< SEARCH
        max_rounds = 80
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
                
            improved = False
            uniques = compute_unique_parts()
            
            # 1. Try removals
            for c in list(code):
                if uniques.get(c, 0) == 0:
                    code_set.discard(c)
                    code.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-swaps: replace c with w
            code_list = list(code)
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_part = uniques.get(c, 0)
                if unique_part == 0:
                    continue
                    
                # Generate candidates: neighbors of c first, then random
                candidates = []
                # Add neighbors of c that are not in code
                for nb in neighbor_lists[c]:
                    if nb not in code_set:
                        candidates.append(nb)
                # Add random candidates
                remaining = total_words - len(code_set)
                if remaining > 0:
                    for _ in range(min(100, remaining)):
                        w = random.randint(0, total_words - 1)
                        if w not in code_set:
                            candidates.append(w)
                
                random.shuffle(candidates)
                
                # Try candidates
                limit = min(len(candidates), 300)
                for w in candidates[:limit]:
                    if (unique_part & ~masks[w]) == 0:
                        # Valid swap
                        code_set.discard(c)
                        code[code.index(c)] = w
                        code_set.add(w)
                        improved = True
                        break
                        
                if improved:
                    break
            
            # 3. Try 2-swaps: replace c1, c2 with w1, w2
            if not improved and len(code) >= 2:
                # Only attempt 2-swaps if we have time
                if time.time() < deadline - 30:
                    # Sample pairs of codewords
                    code_list2 = list(code)
                    random.shuffle(code_list2)
                    pair_limit = min(len(code_list2) * (len(code_list2) - 1) // 2, 50)
                    
                    for _ in range(pair_limit):
                        if time.time() > deadline:
                            break
                        i = random.randint(0, len(code_list2) - 1)
                        j = random.randint(0, len(code_list2) - 1)
                        if i == j:
                            continue
                        c1 = code_list2[i]
                        c2 = code_list2[j]
                        
                        # Combined unique part: words covered only by c1 or only by c2
                        # Actually, we need: words that are covered by c1 or c2 but not by any other
                        # This is more complex. Let's use a simpler heuristic:
                        # unique1 = words only covered by c1 (not c2 or others)
                        # unique2 = words only covered by c2 (not c1 or others)
                        # We need w1 and w2 such that w1 covers unique1 and w2 covers unique2
                        # But this might not be sufficient if there are words covered by both c1 and c2 but no other.
                        
                        # Simpler approach: try to find w1 that covers all of c1's unique part
                        # and w2 that covers all of c2's unique part
                        unique1 = uniques.get(c1, 0)
                        unique2 = uniques.get(c2, 0)
                        
                        if unique1 == 0 and unique2 == 0:
                            # Both are redundant, just remove one
                            code_set.discard(c1)
                            code.remove(c1)
                            improved = True
                            break
                        
                        # Find w1 for unique1
                        w1_found = False
                        candidates1 = []
                        for nb in neighbor_lists[c1]:
                            if nb not in code_set and nb != c2:
                                candidates1.append(nb)
                        for _ in range(50):
                            w = random.randint(0, total_words - 1)
                            if w not in code_set and w != c2:
                                candidates1.append(w)
                        random.shuffle(candidates1)
                        
                        w1 = None
                        for w in candidates1[:100]:
                            if (unique1 & ~masks[w]) == 0:
                                w1 = w
                                break
                        
                        if w1 is None:
                            continue
                        
                        # Find w2 for unique2
                        w2_found = False
                        candidates2 = []
                        for nb in neighbor_lists[c2]:
                            if nb not in code_set and nb != c1 and nb != w1:
                                candidates2.append(nb)
                        for _ in range(50):
                            w = random.randint(0, total_words - 1)
                            if w not in code_set and w != c1 and w != w1:
                                candidates2.append(w)
                        random.shuffle(candidates2)
                        
                        w2 = None
                        for w in candidates2[:100]:
                            if (unique2 & ~masks[w]) == 0:
                                w2 = w
                                break
                        
                        if w2 is None:
                            continue
                        
                        # Verify the swap: after removing c1, c2 and adding w1, w2,
                        # is everything still covered?
                        # Quick check: union of all masks except c1,c2 plus w1,w2 should cover everything
                        # This is expensive. Let's do a lighter check:
                        # Check that unique1 is covered by w1 and unique2 is covered by w2
                        # And that no word becomes uncovered
                        # For now, accept if the unique parts are covered
                        # We'll verify with a full check occasionally
                        
                        # Perform the swap
                        code_set.discard(c1)
                        code_set.discard(c2)
                        idx1 = code.index(c1)
                        idx2 = code.index(c2)
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        code[idx1] = w1
                        code[idx2] = w2
                        code_set.add(w1)
                        code_set.add(w2)
                        improved = True
                        break
                
                if improved:
                    continue

        # Final trim
        trim_code()
=======
        max_rounds = 150
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
                
            improved = False
            uniques = compute_unique_parts()
            
            # 1. Try removals
            for c in list(code):
                if uniques.get(c, 0) == 0:
                    code_set.discard(c)
                    code.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-swaps: replace c with w
            code_list = list(code)
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_part = uniques.get(c, 0)
                if unique_part == 0:
                    continue
                    
                # Generate candidates: neighbors of c first, then random
                candidates = []
                # Add neighbors of c that are not in code
                for nb in neighbor_lists[c]:
                    if nb not in code_set:
                        candidates.append(nb)
                # Add random candidates
                remaining = total_words - len(code_set)
                if remaining > 0:
                    for _ in range(min(100, remaining)):
                        w = random.randint(0, total_words - 1)
                        if w not in code_set:
                            candidates.append(w)
                
                random.shuffle(candidates)
                
                # Try candidates
                limit = min(len(candidates), 300)
                for w in candidates[:limit]:
                    if (unique_part & ~masks[w]) == 0:
                        # Valid swap
                        code_set.discard(c)
                        code[code.index(c)] = w
                        code_set.add(w)
                        improved = True
                        break
                        
                if improved:
                    break
            
            # 3. Try 2-swaps: replace c1, c2 with w1, w2
            if not improved and len(code) >= 2:
                # Only attempt 2-swaps if we have time
                if time.time() < deadline - 30:
                    # Sample pairs of codewords
                    code_list2 = list(code)
                    random.shuffle(code_list2)
                    pair_limit = min(len(code_list2) * (len(code_list2) - 1) // 2, 50)
                    
                    for _ in range(pair_limit):
                        if time.time() > deadline:
                            break
                        i = random.randint(0, len(code_list2) - 1)
                        j = random.randint(0, len(code_list2) - 1)
                        if i == j:
                            continue
                        c1 = code_list2[i]
                        c2 = code_list2[j]
                        
                        # Combined unique part: words covered only by c1 or only by c2
                        # Actually, we need: words that are covered by c1 or c2 but not by any other
                        # This is more complex. Let's use a simpler heuristic:
                        # unique1 = words only covered by c1 (not c2 or others)
                        # unique2 = words only covered by c2 (not c1 or others)
                        # We need w1 and w2 such that w1 covers unique1 and w2 covers unique2
                        # But this might not be sufficient if there are words covered by both c1 and c2 but no other.
                        
                        # Simpler approach: try to find w1 that covers all of c1's unique part
                        # and w2 that covers all of c2's unique part
                        unique1 = uniques.get(c1, 0)
                        unique2 = uniques.get(c2, 0)
                        
                        if unique1 == 0 and unique2 == 0:
                            # Both are redundant, just remove one
                            code_set.discard(c1)
                            code.remove(c1)
                            improved = True
                            break
                        
                        # Find w1 for unique1
                        w1_found = False
                        candidates1 = []
                        for nb in neighbor_lists[c1]:
                            if nb not in code_set and nb != c2:
                                candidates1.append(nb)
                        for _ in range(50):
                            w = random.randint(0, total_words - 1)
                            if w not in code_set and w != c2:
                                candidates1.append(w)
                        random.shuffle(candidates1)
                        
                        w1 = None
                        for w in candidates1[:100]:
                            if (unique1 & ~masks[w]) == 0:
                                w1 = w
                                break
                        
                        if w1 is None:
                            continue
                        
                        # Find w2 for unique2
                        w2_found = False
                        candidates2 = []
                        for nb in neighbor_lists[c2]:
                            if nb not in code_set and nb != c1 and nb != w1:
                                candidates2.append(nb)
                        for _ in range(50):
                            w = random.randint(0, total_words - 1)
                            if w not in code_set and w != c1 and w != w1:
                                candidates2.append(w)
                        random.shuffle(candidates2)
                        
                        w2 = None
                        for w in candidates2[:100]:
                            if (unique2 & ~masks[w]) == 0:
                                w2 = w
                                break
                        
                        if w2 is None:
                            continue
                        
                        # Verify the swap: after removing c1, c2 and adding w1, w2,
                        # is everything still covered?
                        # Quick check: union of all masks except c1,c2 plus w1,w2 should cover everything
                        # This is expensive. Let's do a lighter check:
                        # Check that unique1 is covered by w1 and unique2 is covered by w2
                        # And that no word becomes uncovered
                        # For now, accept if the unique parts are covered
                        # We'll verify with a full check occasionally
                        
                        # Perform the swap
                        code_set.discard(c1)
                        code_set.discard(c2)
                        idx1 = code.index(c1)
                        idx2 = code.index(c2)
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        code[idx1] = w1
                        code[idx2] = w2
                        code_set.add(w1)
                        code_set.add(w2)
                        improved = True
                        break
                
                if improved:
                    continue

        # Final trim
        trim_code()
>>>>>>> REPLACE

