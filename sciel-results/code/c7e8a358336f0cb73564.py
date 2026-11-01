        # 3. Or-opt (relocation of segments) - simplified but effective
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for L in [1, 2, 3]:
                if time.time() > deadline:
                    break
                for i in range(len(order_arr) - L):
                    if time.time() > deadline:
                        break
                    
                    segment = order_arr[i:i + L].copy()
                    
                    # Calculate current cost of removing segment
                    # Edges removed:
                    removed_cost = 0
                    if i > 0:
                        removed_cost += overlap_matrix[order_arr[i - 1]][order_arr[i]]
                    for k in range(i, i + L - 1):
                        removed_cost += overlap_matrix[order_arr[k]][order_arr[k + 1]]
                    if i + L < len(order_arr):
                        removed_cost += overlap_matrix[order_arr[i + L - 1]][order_arr[i + L]]
                    
                    # Edge added when removing: (i-1, i+L) if both exist
                    removed_edge_cost = 0
                    if i > 0 and i + L < len(order_arr):
                        removed_edge_cost = overlap_matrix[order_arr[i - 1]][order_arr[i + L]]
                    
                    # Try inserting at each position
                    best_delta = 0
                    best_pos = -1
                    
                    for pos in range(len(order_arr) - L + 1):
                        if pos == i:
                            continue  # Same position
                        
                        # Adjust pos if it's after the removal point
                        insert_pos = pos
                        if pos > i:
                            insert_pos = pos - L
                        
                        # Calculate cost of inserting at insert_pos
                        # Edges added:
                        added_cost = 0
                        if insert_pos > 0:
                            added_cost += overlap_matrix[order_arr[insert_pos - 1]][segment[0]]
                        for k in range(L - 1):
                            added_cost += overlap_matrix[segment[k]][segment[k + 1]]
                        if insert_pos + L < len(order_arr) - (L if insert_pos >= i else 0):
                            # Need to be careful with indexing after removal
                            # Simplified: just check if there's a next element
                            next_idx = insert_pos + L
                            if next_idx < len(order_arr) - (L if insert_pos >= i else 0):
                                added_cost += overlap_matrix[segment[-1]][order_arr[next_idx]]
                        
                        # Edge removed when inserting: (insert_pos-1, insert_pos) if both exist
                        inserted_edge_cost = 0
                        if insert_pos > 0 and insert_pos + L < len(order_arr) - (L if insert_pos >= i else 0):
                            inserted_edge_cost = overlap_matrix[order_arr[insert_pos - 1]][order_arr[insert_pos]]
                        
                        # Net change in overlap
                        # We remove 'removed_cost' of overlap edges and add 'removed_edge_cost'
                        # We add 'added_cost' of overlap edges and remove 'inserted_edge_cost'
                        # Net delta in overlap = (removed_edge_cost - removed_cost) + (added_cost - inserted_edge_cost)
                        delta = (removed_edge_cost - removed_cost) + (added_cost - inserted_edge_cost)
                        
                        if delta > best_delta:
                            best_delta = delta
                            best_pos = pos
                    
                    if best_delta > 0 and best_pos >= 0:
                        # Perform the move
                        remaining = np.concatenate([order_arr[:i], order_arr[i + L:]])
                        insert_pos = best_pos
                        if insert_pos > i:
                            insert_pos = insert_pos - L
                        new_order = np.concatenate([remaining[:insert_pos], segment, remaining[insert_pos:]])
                        order_arr = new_order
                        best_length -= best_delta
                        best_order = order_arr.tolist()
                        improved = True
