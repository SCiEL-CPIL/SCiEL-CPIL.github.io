   def _get_hex_packing():
       # Hexagonal pattern for 21 circles: 5-4-5-4-3
       # W = 10r, H = (4*sqrt(3) + 1)r
       W_unit = 10.0
       H_unit = 4 * np.sqrt(3) + 1.0
       
       # Optimal aspect ratio for container to match pattern aspect
       # w/h = W_unit/H_unit
       # w + h = 2
       # h = 2 / (1 + W_unit/H_unit) = 2 * H_unit / (W_unit + H_unit)
       # w = 2 * W_unit / (W_unit + H_unit)
       
       w = 2.0 * W_unit / (W_unit + H_unit)
       h = 2.0 * H_unit / (W_unit + H_unit)
       
       # Scale factor s such that s * W_unit = w (and s * H_unit = h)
       s = w / W_unit
       
       r = s * 1.0 # Since W_unit was defined for r=1
       
       # Generate centers
       centers = []
       # Row 0: 5 circles
       for i in range(5):
           centers.append((i * 2.0 * r, 0.0))
       # Row 1: 4 circles
       for i in range(4):
           centers.append((r + i * 2.0 * r, np.sqrt(3) * r))
       # Row 2: 5 circles
       for i in range(5):
           centers.append((i * 2.0 * r, 2.0 * np.sqrt(3) * r))
       # Row 3: 4 circles
       for i in range(4):
           centers.append((r + i * 2.0 * r, 3.0 * np.sqrt(3) * r))
       # Row 4: 3 circles
       for i in range(3):
           centers.append((2.0 * r + i * 2.0 * r, 4.0 * np.sqrt(3) * r))
           
       centers = np.array(centers)
       
       # The pattern currently spans from x=0 to x=8r (width 8r)?? 
       # Wait, my W_unit calculation:
       # Row 0: x=0, 2r, 4r, 6r, 8r. Max x = 8r. Min x = 0.
       # But circles have radius r. So min x extent = -r, max x extent = 9r.
       # Width = 10r. Correct.
       
       # The pattern y extent:
       # Min y = 0 - r = -r.
       # Max y = 4*sqrt(3)r + r.
       # Height = (4*sqrt(3) + 1)r. Correct.
       
       # We need to center this pattern in the w x h box.
       # Current center of pattern:
       # x_center_pattern = ( -r + 9r ) / 2 = 4r
       # y_center_pattern = ( -r + (4*sqrt(3)+1)r ) / 2 = 2*sqrt(3)r
       
       # Box center:
       # x_center_box = w / 2
       # y_center_box = h / 2
       
       # Shift
       dx = w / 2.0 - 4.0 * r
       dy = h / 2.0 - 2.0 * np.sqrt(3) * r
       
       centers[:, 0] += dx
       centers[:, 1] += dy
       
       radii = np.full(21, r)
       
       circles = np.column_stack((centers, radii))
       return circles, w, h
   