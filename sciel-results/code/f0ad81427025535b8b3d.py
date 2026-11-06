            if stagnation > max_stagnation:
                # Random kick: change 5 random residues
                for _ in range(5):
                    i = random.randint(0, num_divs - 1)
                    m = divisors[i]
                    new_r = random.randint(0, m - 1)
                    old_r = current_residues[i]
                    if new_r != old_r:
                        old_idx = indices[m][old_r]
                        new_idx = indices[m][new_r]
                        current_cover_count[old_idx] -= 1
                        current_cover_count[new_idx] += 1
                        current_residues[i] = new_r
                        # Recalculate uncovered for this change?
                        # This is messy to do incrementally for multiple kicks.
                        # Easier: Just reset to best and let it search again.
                # Let's just reset to best.
                current_residues = best_residues[:]
                current_cover_count = best_cover_count.copy()
                current_uncovered = best_uncovered
                stagnation = 0
                max_stagnation = int(max_stagnation * 1.5)
