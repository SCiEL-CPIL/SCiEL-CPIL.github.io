   import numpy as np

   def construct():
       N, M, P = 2, 3, 10
       U = np.ones((N * M, 1))
       V = np.ones((M * P, 1))
       W = np.ones((P * N, 1))
       return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
   