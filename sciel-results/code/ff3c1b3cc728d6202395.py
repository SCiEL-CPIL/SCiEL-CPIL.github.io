<<<<<<< SEARCH
                # Find best replacement: word that covers most of the uncovered
                # and doesn't duplicate existing codewords
                existing_set = set(remaining)
                best_new = -1
                best_count = 0
                
                for cand in uncovered:
                    # Count how many uncovered words this candidate covers
                    cnt = int(cover_mask[cand, uncovered].sum())
                    if cnt > best_count:
                        best_count = cnt
                        best_new = cand
                
                # Also check all words, not just uncovered ones, for a better swap
                for cand in range(N):
                    if all_words[cand].tolist() in existing_set:
                        continue
                    cnt = int(cover_mask[cand, uncovered].sum())
                    if cnt > best_count:
                        best_count = cnt
                        best_new = cand
=======
                # Find best replacement: word that covers most of the uncovered
                # and doesn't duplicate existing codewords
                existing_set = set(tuple(w) for w in remaining)
                best_new = -1
                best_count = 0
                
                # Vectorized count of coverage of 'uncovered' by all candidates
                if len(uncovered) > 0:
                    counts = cover_mask[:, uncovered].sum(axis=1)
                else:
                    counts = np.zeros(N, dtype=np.int32)
                
                for cand in range(N):
                    if tuple(all_words[cand]) in existing_set:
                        continue
                    if counts[cand] > best_count:
                        best_count = counts[cand]
                        best_new = cand
>>>>>>> REPLACE
