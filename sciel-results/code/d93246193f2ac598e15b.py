import numpy as np

def construct():
       n = 16
       # Construct K8,8 adjacency matrix
       A = np.zeros((n, n), dtype=int)
       for i in range(8):
           for j in range(8, 16):
               A[i, j] = 1
               A[j, i] = 1
               
       edges = []
       for i in range(n):
           for j in range(i+1, n):
               if A[i, j] == 1:
                   edges.append([i, j])
       return edges
