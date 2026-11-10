        covered_count = 0
        while covered_count < N:
            # ...
            for j in neighbors[best_idx]:
                if not covered[j]:
                    covered[j] = True
                    covered_count += 1
