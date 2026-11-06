import numpy as np
import time

def construct():
       N = 18
       MIN_DIST = 6
       
       # 1. Try Extended QR Code Construction
       try:
           # Quadratic residues mod 17
           residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
           v = np.zeros(17, dtype=np.uint8)
           v[residues] = 1
           
           G = np.zeros((9, 18), dtype=np.uint8)
           for i in range(9):
               row = np.roll(v, i)
               G[i, :17] = row
               G[i, 17] = row.sum() % 2
               
           # Generate codewords
           codes = []
           for m in range(512):
               c = 0
               for j in range(9):
                   if (m >> j) & 1:
                       # XOR row j into c
                       # Convert G[j] to int
                       # G[j] is uint8 array
                       # We can precompute int versions
                       pass
               
           # Precompute int versions of rows for speed
           row_ints = []
           for j in range(9):
               val = 0
               for k in range(18):
                   if G[j, k]:
                       val |= (1 << k)
               row_ints.append(val)
               
           for m in range(512):
               c = 0
               for j in range(9):
                   if (m >> j) & 1:
                       c ^= row_ints[j]
               codes.append(c)
               
           # Verify
           min_d = N + 1
           for i in range(len(codes)):
               for j in range(i+1, len(codes)):
                   d = (codes[i] ^ codes[j]).bit_count()
                   if d < min_d:
                       min_d = d
                   if min_d < MIN_DIST:
                       break
               if min_d < MIN_DIST:
                   break
           
           if min_d >= MIN_DIST:
               return codes
           else:
               # Fallback
               pass
       except Exception:
           pass
       
       # Fallback: Optimized Greedy
       # ...
