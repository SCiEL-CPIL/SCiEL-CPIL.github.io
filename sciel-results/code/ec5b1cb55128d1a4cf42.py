"""Size-efficient sorting networks on 16 wires."""

N = 16

# EVOLVE-BLOCK-START
def construct():
    """Return the odd-even transposition sorting network for 16 wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
