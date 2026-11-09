<<<<<<< SEARCH
import random
import time
import numpy as np

N = 64
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
# dist_matrix[i, j] = squared distance between POINTS[i] and POINTS[j]
dist_matrix = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d2 = (xi - xj) ** 2 + (yi - yj) ** 2
        dist_matrix[i, j] = d2
        dist_matrix[j, i] = d2
=======
import random
import time
import numpy as np

N = 64
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances using vectorized NumPy
xs = np.array([p[0] for p in POINTS], dtype=np.int32)
ys = np.array([p[1] for p in POINTS], dtype=np.int32)
diff_x = xs[:, None] - xs[None, :]
diff_y = ys[:, None] - ys[None, :]
dist_matrix = (diff_x ** 2 + diff_y ** 2).astype(np.int32)
>>>>>>> REPLACE

