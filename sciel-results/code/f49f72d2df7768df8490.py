        S = set(candidate)
        score = 0
        for a in S:
            for d in range(1, (N - a) // 2 + 1):
                if a + d in S and a + 2*d in S:
                    score += 1
