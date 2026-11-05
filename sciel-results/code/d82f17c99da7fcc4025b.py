def construct():
       n = 40
       edges = []
       # Hamiltonian cycle
       for i in range(n):
           edges.append((i, (i + 1) % n))
       
       # Chords (0, k) for k in 2..20
       for k in range(2, 21):
           edges.append((0, k))
           
       # Sort edges for consistency (optional but good)
       edges = sorted(set(tuple(sorted(e)) for e in edges))
       
       cycles = []
       for L in range(3, n + 1):
           if L <= 20:
               # Cycle 0-1-...-(L-1)-0
               cycles.append(list(range(L)))
           elif L == 21:
               # Cycle 0-1-...-20-0
               cycles.append(list(range(21)))
           elif L <= 39:
               # s = 41 - L
               # Cycle 0-39-...-(41-L)-0
               start = 39
               end = 41 - L
               # Vertices: 0, 39, 38, ..., end
               verts = [0] + list(range(start, end - 1, -1))
               cycles.append(verts)
           else: # L == 40
               cycles.append(list(range(n)))
               
       return {'edges': edges, 'cycles': cycles}
