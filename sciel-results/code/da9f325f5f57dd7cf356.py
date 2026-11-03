            for c in candidates:
                cnt = 0
                for w in coverage[c]:
                    if w in uncovered:
                        cnt += 1
                if cnt > max_covered:
                    max_covered = cnt
                    best_c = c
