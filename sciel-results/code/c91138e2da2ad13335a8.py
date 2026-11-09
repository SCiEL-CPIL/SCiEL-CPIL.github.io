"""Rewritten program: 2x15 grid to maximize equal-area triangle multiplicity."""
# EVOLVE-BLOCK-START
def construct():
    pts = []
    # 2 rows, 15 columns
    for i in range(15):
        pts.append((i, 0))
        pts.append((i, 1))
    return pts
# EVOLVE-BLOCK-END
