  def construct():
      N = 40
      edges = []
      for i in range(N):
          edges.append((i, (i + 1) % N))
      for k in range(10, 21):
          edges.append((0, k))
      
      cycles = []
      for L in range(3, 41):
          if 3 <= L <= 12:
              d = L - 2
              # i=10, j=10+d
              # Cycle: 0, 10, 11, ..., 10+d
              cycle = [0] + list(range(10, 10 + d + 1))
          elif 11 <= L <= 20:
              # k=L
              cycle = list(range(L + 1))
          elif 21 <= L <= 31:
              # k=41-L
              k = 41 - L
              cycle = [0] + list(range(k, 40))
          else: # 32 <= L <= 40
              d = 42 - L
              # i=10, j=10-d
              # Path 10 -> 9 -> ... -> 10-d
              cycle = [0] + list(range(10, 10 - d - 1, -1))
          
          cycles.append(cycle)
          
      return {'edges': edges, 'cycles': cycles}
  