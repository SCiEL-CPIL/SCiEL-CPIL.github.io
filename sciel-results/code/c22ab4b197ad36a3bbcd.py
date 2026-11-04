    def neighbours(vertex):
        """Generate neighbors of a vertex in the LUW graph."""
        side, x = vertex
        # x is a tuple of length d+1: (None, x1, x2, ..., xd)
        # side 0 = point, side 1 = line
        
        x1, x2, x3, x4, x5 = x[1], x[2], x[3], x[4], x[5]
        
        for first in range(q):
            # y[1] = first
            # y[2] depends on rule[2] = (1, 1)
            # y[3] depends on rule[3] = (2, 1)
            # y[4] depends on rule[4] = (1, 2)
            # y[5] depends on rule[5] = (1, 3)
            
            y1 = first
            
            if side == 0:
                # y[c] = x[c] + y[a] * x[b]
                y2 = gf4_add(x2, gf4_mul(y1, x1))
                y3 = gf4_add(x3, gf4_mul(y2, x1))
                y4 = gf4_add(x4, gf4_mul(y1, x2))
                y5 = gf4_add(x5, gf4_mul(y1, x3))
            else:
                # y[c] = x[c] - x[a] * y[b]
                y2 = gf4_sub(x2, gf4_mul(x1, y1))
                y3 = gf4_sub(x3, gf4_mul(x2, y1))
                y4 = gf4_sub(x4, gf4_mul(x1, y2))
                y5 = gf4_sub(x5, gf4_mul(x1, y3))
            
            yield (1 - side), (None, y1, y2, y3, y4, y5)
