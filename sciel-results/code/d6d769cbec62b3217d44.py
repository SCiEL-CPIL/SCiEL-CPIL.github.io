   dx = vx - Cx
   dy = vy - Cy
   projs = np.abs(dx * normals[:, 0] + dy * normals[:, 1])
   