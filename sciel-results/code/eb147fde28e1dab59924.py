def construct():
       points = []
       # Line y=0
       for x in range(8):
           points.append((x, 0))
       # Line y=1
       for x in range(14):
           points.append((x, 1))
       # Line y=2
       for x in range(8):
           points.append((x, 2))
       return points
