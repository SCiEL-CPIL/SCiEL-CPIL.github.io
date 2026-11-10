            best_idx = -1
            best_count = -1
            candidates = []
            
            for i in range(num_words):
                count = 0
                for n_idx in neighbors[i]:
                    if n_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
