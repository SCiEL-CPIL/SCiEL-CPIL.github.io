<<<<<<< SEARCH
        cand = []
        unsatisfied = [t for t in targets if need.get(t, 10**9) > 1]
        if not unsatisfied:
            break
        for c, (a, b) in pool.items():
            score = 0.0
            for t in unsatisfied:
                wt = bin(t).count('1')
                wt2 = bin(t ^ c).count('1')
                if wt2 < wt:
                    # Weight by urgency: targets closer to completion get higher weight
                    urgency = 1.0 / (1.0 + wt)
                    score += (wt - wt2) * urgency
            if score > 0:
                cand.append((score, c, a, b))
=======
        cand = []
        unsatisfied = [t for t in unique_targets if need.get(t, 10**9) > 1]
        if not unsatisfied:
            break
        for c, (a, b) in pool.items():
            score = 0.0
            for t in unsatisfied:
                wt = bin(t).count('1')
                wt2 = bin(t ^ c).count('1')
                if wt2 < wt:
                    # Weight by urgency: targets closer to completion get higher weight
                    urgency = 1.0 / (1.0 + wt)
                    score += (wt - wt2) * urgency
            if score > 0:
                cand.append((score, c, a, b))
>>>>>>> REPLACE

