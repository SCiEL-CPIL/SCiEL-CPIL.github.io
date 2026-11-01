# Inside n==7 block
        # ... after greedy_order is computed and score updated ...
        
        # Local Search on the best order found so far
        # Use adjacent swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(order) - 1):
                if time.time() > deadline:
                    break
                # Swap order[i] and order[i+1]
                idx1 = order[i]
                idx2 = order[i+1]
                
                # Calculate delta
                # Old edges: (i-1, i), (i, i+1), (i+1, i+2)
                # New edges: (i-1, i+1), (i+1, i), (i, i+2)
                
                old_score_part = 0
                new_score_part = 0
                
                if i > 0:
                    old_score_part += M[order[i-1]][idx1]
                    new_score_part += M[order[i-1]][idx2]
                
                old_score_part += M[idx1][idx2]
                new_score_part += M[idx2][idx1]
                
                if i < len(order) - 2:
                    old_score_part += M[idx2][order[i+2]]
                    new_score_part += M[idx1][order[i+2]]
                
                delta = new_score_part - old_score_part
                if delta > 0:
                    order[i], order[i+1] = order[i+1], order[i]
                    score += delta
                    improved = True
        
        # Reconstruct word
        final_word = []
        for k, idx in enumerate(order):
            p = perms_6[idx]
            block = list(p) + [7] + list(p)
            if k == 0:
                final_word = block
            else:
                prev_idx = order[k-1]
                ov = M[prev_idx][idx]
                final_word += block[ov:]
        
        return "".join(map(str, final_word))
