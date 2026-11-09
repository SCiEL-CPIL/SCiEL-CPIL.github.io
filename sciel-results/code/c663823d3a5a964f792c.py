"""Optimized program: 3 parallel lines (8-14-8) to maximize equal-area triangle multiplicity."""
# EVOLVE-BLOCK-START
def construct():
    pts = []
    # Line 0: y=0, x=0..7
    for x in range(8):
        pts.append((x, 0))
    # Line 1: y=1, x=0..13
    for x in range(14):
        pts.append((x, 1))
    # Line 2: y=2, x=0..7
    for x in range(8):
        pts.append((x, 2))
    return pts
# EVOLVE-BLOCK-END
