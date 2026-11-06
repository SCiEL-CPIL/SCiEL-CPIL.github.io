        for j in range(i + 1, N):
            if _is_square(val + (j + 1)):
                m |= (1 << j)
        masks[i] = m
