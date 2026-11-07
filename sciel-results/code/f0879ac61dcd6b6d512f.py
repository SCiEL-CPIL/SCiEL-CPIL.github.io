        for u, v in remaining:
            common = 0
            for w in range(n):
                if w != u and w != v:
                    if (min(u, w), max(u, w)) in remaining and (min(v, w), max(v, w)) in remaining:
                        common += 1
            score = common
            if score > best_score:
                best_score = score
                best_edge = (u, v)
