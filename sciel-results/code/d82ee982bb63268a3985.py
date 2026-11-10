            if new_mask == old_mask or new_mask in used:
                continue
                
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                m_j = masks[j]
                old_c = comp[old_mask][m_j]
                new_c = comp[new_mask][m_j]
                delta += new_c - old_c
            
            if delta > 0 or random.random() < (2.71828 ** (delta / T)):
                used.remove(old_mask)
                used.add(new_mask)
                masks[i] = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_masks = masks[:]
                T *= cooling
