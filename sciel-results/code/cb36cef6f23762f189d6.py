def construct():
       p = 1009
       roots = {}
       for x in range(1, p):
           r = (x * x) % p
           if r not in roots:
               roots[r] = x
       
       for s in range(1, p - 4): # s up to 1004
           # Check if s, s+1, s+2, s+3, s+4 are in roots
           # Since s+i <= 1008, they are non-zero.
           if all((s + i) in roots for i in range(5)):
               return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
       
       raise RuntimeError('No run found')
