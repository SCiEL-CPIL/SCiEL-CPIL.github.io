<<<<<<< SEARCH
def _full_counter(pts):
    n = len(pts)
    pts_arr = np.array(pts, dtype=np.int64)
    # Compute all triple determinants using broadcasting
    # For triples (i,j,k) with i<j<k
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts_arr[j,0]-pts_arr[i,0]) * int(pts_arr[k,1]-pts_arr[i,1]) - 
                          int(pts_arr[j,1]-pts_arr[i,1]) * int(pts_arr[k,0]-pts_arr[i,0]))
                if det > 0:
                    areas.append(det)
    return Counter(areas)
=======
def _full_counter(pts):
    n = len(pts)
    areas = []
    for i in range(n):
        ax, ay = pts[i]
        for j in range(i+1, n):
            bx, by = pts[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = pts[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0:
                    areas.append(det)
    return Counter(areas)
>>>>>>> REPLACE
