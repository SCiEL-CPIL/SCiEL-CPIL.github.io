        # Current:
        for v in Available:
            new_bad = 0
            for u in S_list: new_bad |= bad_thirds[u][v]
            removal = (Available & new_bad).bit_count()
        