        best_idx = -1
        best_gain = -1
        candidates = []
        
        for i in range(num_k):
            if selected_mask[i]:
                continue
            gain = (uncovered & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                candidates = [i]
            elif gain == best_gain:
                candidates.append(i)
        
        if best_idx == -1 or best_gain == 0:
            break
            
        best_idx = random.choice(candidates)
