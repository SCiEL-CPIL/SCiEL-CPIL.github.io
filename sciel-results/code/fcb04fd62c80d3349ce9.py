            best_gain = -1
            best_candidates = []
            
            for b_idx in candidates:
                covered_by_b = block_to_mask[b_idx] & uncovered_mask
                gain = covered_by_b.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
            
            # Pick randomly from best candidates
            best_block_idx = random.choice(best_candidates)
