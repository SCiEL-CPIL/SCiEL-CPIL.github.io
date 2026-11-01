<<<<<<< SEARCH
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
=======
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y0 = [0] * (d + 1)
                y0[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y0[c] = add(x[c], mul(y0[a], x[b]))
                    else:
                        y0[c] = sub(x[c], mul(x[a], y0[b]))
                yield 1 - side, tuple(y0)
>>>>>>> REPLACE
