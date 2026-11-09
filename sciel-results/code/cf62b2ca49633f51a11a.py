   for L in range(3, N+1):
       if L <= 22:
           d = L - 1
           cycle = list(range(0, d+1))
       else:
           d = 41 - L
           cycle = [0] + list(range(39, d-1, -1))
   