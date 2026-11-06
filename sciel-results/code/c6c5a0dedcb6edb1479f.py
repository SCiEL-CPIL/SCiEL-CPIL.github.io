def construct():
         n = 32
         edges = []
         for b in range(8):
             start = b * 4
             for i in range(4):
                 for j in range(i + 1, 4):
                     edges.append([start + i, start + j])
         return {'n': n, 'edges': edges}
