<<<<<<< SEARCH
    def try_local_search(cur, bitset, length, deadline):
        """Try to find a larger valid subset by local search with efficient bitset updates."""
        cur_set = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in cur (from largest to smallest)
            for x in range(MAX_ELE, 0, -1):
                if x in cur_set:
                    continue
                if can_add(bitset, x):
                    cur.append(x)
                    cur_set.add(x)
                    length += 1
                    bitset = add_element(bitset, x)
                    improved = True
                    break
            
            if time.time() > deadline - 3.0:
                break
            
            # Try 1-removal + re-add swaps
            for i in range(len(cur)):
                if time.time() > deadline - 3.0:
                    break
                removed = cur[i]
                # Compute bitset without element at position i
                # Use prefix/suffix approach for efficiency
                # Recompute from scratch (n is at most ~50, so this is fast)
                bitset_no = 1
                for j, x in enumerate(cur):
                    if j == i:
                        continue
                    bitset_no = add_element(bitset_no, x)
                
                # Try to find a better element to swap in
                best_add = None
                best_val = removed
                for x in range(MAX_ELE, 0, -1):
                    if x in cur_set:
                        continue
                    if can_add(bitset_no, x):
                        if x > best_val:
                            best_add = x
                            best_val = x
                
                if best_add is not None and best_add != removed:
                    cur[i] = best_add
                    cur_set.discard(removed)
                    cur_set.add(best_add)
                    bitset = add_element(bitset_no, best_add)
                    improved = True
        
        return cur, bitset, length
=======
    def try_local_search(cur, bitset, length, deadline):
        """Try to find a larger valid subset by local search with efficient bitset updates."""
        cur_set = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in cur (from largest to smallest)
            for x in range(MAX_ELE, 0, -1):
                if x in cur_set:
                    continue
                if can_add(bitset, x):
                    cur.append(x)
                    cur_set.add(x)
                    length += 1
                    bitset = add_element(bitset, x)
                    improved = True
                    break
            
            if time.time() > deadline - 3.0:
                break
            
            # Try 1-removal + multi-add: remove one element, then greedily add as many as possible
            best_gain = 0
            best_idx = -1
            best_new_cur = None
            best_new_bitset = None
            
            for i in range(len(cur)):
                if time.time() > deadline - 3.0:
                    break
                removed = cur[i]
                # Compute bitset without element at position i
                bitset_no = 1
                for j, x in enumerate(cur):
                    if j == i:
                        continue
                    bitset_no = add_element(bitset_no, x)
                
                # After removing, try to add back as many elements as possible
                new_cur = list(cur)
                new_cur.pop(i)
                new_cur_set = set(new_cur)
                new_bitset = bitset_no
                gain = 0
                
                for x in range(MAX_ELE, 0, -1):
                    if x in new_cur_set:
                        continue
                    if can_add(new_bitset, x):
                        new_cur.append(x)
                        new_cur_set.add(x)
                        new_bitset = add_element(new_bitset, x)
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_new_cur = new_cur
                    best_new_bitset = new_bitset
            
            if best_gain > 0 and best_new_cur is not None:
                cur = best_new_cur
                cur_set = set(cur)
                length = len(cur)
                bitset = best_new_bitset
                improved = True
        
        return cur, bitset, length
>>>>>>> REPLACE

