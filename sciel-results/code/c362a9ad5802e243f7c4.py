     import numpy as np
     import time

     N = 11
     SQRT3 = np.sqrt(3)
     H = SQRT3 / 2.0

     def get_areas(points):
         # Returns array of 165 signed areas
         n = points.shape[0]
         areas = np.zeros((n, n, n))
         # Vectorized calculation is tricky for all triples, but 165 is small.
         # We can use broadcasting.
         # p1, p2, p3
         # Area = 0.5 * | (p2-p1) x (p3-p1) |
         # Let's just loop for clarity and speed (165 is tiny).
         res = []
         for i in range(n):
             for j in range(i+1, n):
                 for k in range(j+1, n):
                     p1 = points[i]
                     p2 = points[j]
                     p3 = points[k]
                     # Signed area * 2
                     a = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                     res.append(a)
         return np.array(res)

     def get_grad_min_area(points, areas):
         # Returns gradient of the minimum absolute area?
         # Or minimum signed area?
         # Let's assume we want to maximize the minimum *positive* area.
         # If a triangle has negative area, its "size" is -area.
         # To maximize min size, we need to push all triangles to be large.
         
         # Let's find the triangle with the smallest absolute area.
         abs_areas = np.abs(areas)
         min_idx = np.argmin(abs_areas)
         
         # Get the indices i, j, k for min_idx
         # We need a mapping from flat index to (i,j,k)
         # Pre-compute this mapping outside? Or inside?
         # For N=11, 165 items.
         idxs = []
         count = 0
         for i in range(N):
             for j in range(i+1, N):
                 for k in range(j+1, N):
                     if count == min_idx:
                         idxs = (i, j, k)
                         break
                     count += 1
                     if count == min_idx: break
             if count == min_idx: break
         if count != min_idx:
             # Fallback
             return np.zeros((N, 2))
             
         i, j, k = idxs
         p1 = points[i]
         p2 = points[j]
         p3 = points[k]
         
         # Gradient of signed area S = (p2-p1) x (p3-p1)
         # dS/dp1 = - (p3-p2) rotated 90 deg?
         # S = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
         # dS/dx1 = y2-y3, dS/dy1 = x3-x2
         # dS/dx2 = y3-y1, dS/dy2 = x1-x3
         # dS/dx3 = y1-y2, dS/dy3 = x2-x1
         
         grad = np.zeros((N, 2))
         # We want to increase the area.
         # If signed area is positive, we increase S.
         # If signed area is negative, we decrease S (to make it more negative? No, we want |S| to increase).
         # Actually, if we want to maximize |S|, and S < 0, we want to make S more negative (decrease S).
         # So direction is sign(S) * grad_S.
         
         s = areas[min_idx]
         sign = 1.0 if s >= 0 else -1.0
         
         grad[i, 0] += sign * (p2[1] - p3[1])
         grad[i, 1] += sign * (p3[0] - p2[0])
         
         grad[j, 0] += sign * (p3[1] - p1[1])
         grad[j, 1] += sign * (p1[0] - p3[0])
         
         grad[k, 0] += sign * (p1[1] - p2[1])
         grad[k, 1] += sign * (p2[0] - p1[0])
         
         return grad

     def boundary_penalty_grad(points):
         # Gradient of penalty for being outside triangle
         # Triangle: (0,0), (1,0), (0.5, H)
         # Inequalities:
         # y >= 0
         # y <= sqrt(3) x  => sqrt(3)x - y >= 0
         # y <= -sqrt(3)(x-1) => sqrt(3)(1-x) - y >= 0  => sqrt(3) - sqrt(3)x - y >= 0
         
         pen_grad = np.zeros_like(points)
         for i in range(N):
             x, y = points[i]
             # y < 0
             if y < 0:
                 pen_grad[i, 1] += 1.0
             # sqrt(3)x - y < 0
             if SQRT3*x - y < 0:
                 pen_grad[i, 0] += -SQRT3
                 pen_grad[i, 1] += 1.0
             # sqrt(3) - sqrt(3)x - y < 0
             if SQRT3 - SQRT3*x - y < 0:
                 pen_grad[i, 0] += -SQRT3
                 pen_grad[i, 1] += 1.0
         return pen_grad
     