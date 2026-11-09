<<<<<<< SEARCH
                for x in candidates:
                    new_block = block + [x]
                    # Count how many uncovered t-subsets are covered by new_block
                    gain = 0
                    for sub in itertools.combinations(new_block, t):
                        if uncovered[subset_to_idx[sub]]:
                            gain += 1
=======
                for x in candidates:
                    new_block = block + [x]
                    # Count how many uncovered t-subsets are covered by new_block
                    gain = 0
                    for sub in itertools.combinations(new_block, t):
                        # Ensure sub is sorted to match keys in subset_to_idx
                        if uncovered[subset_to_idx[tuple(sorted(sub))]]:
                            gain += 1
>>>>>>> REPLACE
