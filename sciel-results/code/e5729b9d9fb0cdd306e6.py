"""Deterministic construction: 6x6 grid minus a diagonal pair.

The 6x6 integer grid {0..5}^2 has 12 lines of 6 points (6 rows + 6 columns)
plus the two main diagonals (also 6 points). Removing the diagonal pair
(2,2) and (3,3) removes exactly one point from each of the 12 rows/columns
(-> 5 points each) and two points from each main diagonal (-> 4 points each).
No other line contains 5 or more points, so no five are collinear and exactly
14 lines contain exactly four points (the 12 rows/columns + 2 diagonals).
"""
# EVOLVE-BLOCK-START
def construct():
    pts = []
    for x in range(6):
        for y in range(6):
            if (x, y) in ((2, 2), (3, 3)):
                continue
            pts.append([x, y])
    return pts
# EVOLVE-BLOCK-END
