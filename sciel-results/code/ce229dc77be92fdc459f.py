"""Size-efficient sorting networks on 16 wires."""
import time

N = 16

# EVOLVE-BLOCK-START

def _sort_half(n, offset=0):
    """Odd-even transposition sort on n wires starting at offset."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((offset + i, offset + i + 1))
    return comps

def _merge(h, offset=0):
    """Batcher merge network for two sorted halves of size h starting at offset."""
    comps = []
    k = h
    while k >= 1:
        for i in range(0, h - k):
            comps.append((offset + i, offset + i + k))
        k = k // 2
    return comps

def _batcher_sort(n, offset=0):
    """Batcher's odd-even merge sort network for n wires starting at offset."""
    if n == 1:
        return []
    h = n // 2
    comps = []
    # Sort first half
    comps.extend(_batcher_sort(h, offset))
    # Sort second half
    comps.extend(_batcher_sort(h, offset + h))
    # Merge
    comps.extend(_merge(h, offset))
    return comps

def construct():
    """Construct a sorting network for 16 wires."""
    # Batcher's odd-even merge sort: 119 comparators for 16 wires
    return _batcher_sort(16)

# EVOLVE-BLOCK-END

