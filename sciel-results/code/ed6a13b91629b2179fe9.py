            # Inside greedy_construct_fast loop
            uncovered_count = sum(uncovered)
            if len(code) + (uncovered_count + 16) // 17 > best_size:
                break
