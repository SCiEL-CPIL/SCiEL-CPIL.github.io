W_2x2 = np.array([
    [1, 0, 0, 1],    # t=0
    [0, 0, 1, -1],   # t=1
    [0, 1, 0, 1],    # t=2  ← WRONG, should be [0,0,-1,0]
    [1, 0, 1, 0],    # t=3  ← WRONG, should be [-1,0,0,0]
    [-1, 1, 0, 0],   # t=4  ← WRONG, should be [0,1,0,1]
    [0, 0, 0, 1],    # t=5  ← WRONG, should be [1,0,1,0]
    [1, 0, 0, 0]     # t=6  ← WRONG, should be [0,1,1,0]
])
