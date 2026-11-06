        a, b = (i, i + half) if direction == 0 else (i + half, i)
        if a > b:
            a, b = b, a
        comparators.append((a, b))
