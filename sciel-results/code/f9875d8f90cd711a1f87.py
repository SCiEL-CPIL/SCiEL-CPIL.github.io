"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # A = {17..32}, B = {65..128}
    # This set is valid because:
    # 1. Sums within A are in [35, 63], which is disjoint from A and B.
    # 2. Sums within B are > 128, disjoint from S.
    # 3. Sums of two from A are not in S.
    # 4. Sums of two from B are not in S.
    A = list(range(17, 33))
    B = list(range(65, 129))
    return A + B
# EVOLVE-BLOCK-END
