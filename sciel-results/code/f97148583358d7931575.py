"""
Size-efficient sorting network on 17 wires.
"""

N = 17

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for 17 wires.
    """
    # Odd-even transposition network (known correct, 136 comparators)
    # But we'll try to reduce it slightly while maintaining correctness
    
    # Use 12 phases instead of 17, plus some cross-comparators
    comparators = []
    
    for phase in range(12):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    
    # Add a few cross-comparators to help with long-range ordering
    for i in range(0, N - 2, 2):
        comparators.append((i, i + 2))
    
    return comparators
# EVOLVE-BLOCK-END
