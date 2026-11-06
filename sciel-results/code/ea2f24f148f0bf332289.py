"""Size-efficient sorting network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Bitonic sorting network for 16 wires.
    # A bitonic sort of N=16 elements uses (N/2) * log2(N) * (log2(N)+1) / 2 comparators?
    # Actually, the standard bitonic network for N=16 has 120 comparators (same as odd-even transposition).
    # Let's verify: bitonic_sort(16) 
    # Stages: log2(16) = 4 stages of size merging.
    # For a bitonic network of size N, the number of comparators is N/2 * log2(N) * (log2(N)+1)/2? No.
    # Let me count:
    # k=1: merge two halves of size 8? No, the bitonic sort algorithm:
    # bitonic_sort(n): 
    #   if n == 2: return [(0,1)]
    #   c1 = bitonic_sort(n/2) (left half, sorted)
    #   c2 = bitonic_sort(n/2) (right half, sorted)
    #   c3 = bitonic_merge(n, offset)
    #   return c1 + c2 + c3
    # bitonic_merge(n, offset):
    #   if n == 2: return [(offset, offset+1)]
    #   c1 = bitonic_merge(n/2, offset)
    #   c2 = bitonic_merge(n/2, offset + n/2)
    #   c3 = cross-comparators: for i in range(n/2): (offset+i, offset+n/2+i)
    #   return c1 + c2 + c3
    #
    # Count for n=16:
    # bitonic_merge(16, 0):
    #   bitonic_merge(8, 0) + bitonic_merge(8, 8) + 8 cross = 2*M(8) + 8
    # bitonic_merge(8, 0):
    #   bitonic_merge(4, 0) + bitonic_merge(4, 4) + 4 cross = 2*M(4) + 4
    # bitonic_merge(4, 0):
    #   bitonic_merge(2, 0) + bitonic_merge(2, 2) + 2 cross = 2*M(2) + 2
    # bitonic_merge(2, 0): 1 comparator
    # So M(2)=1, M(4)=2*1+2=4, M(8)=2*4+4=12, M(16)=2*12+8=32
    # bitonic_sort(16): 2*S(8) + M(16) = 2*S(8) + 32
    # S(8) = 2*S(4) + M(8) = 2*S(4) + 12
    # S(4) = 2*S(2) + M(4) = 2*1 + 4 = 6
    # S(8) = 2*6 + 12 = 24
    # S(16) = 2*24 + 32 = 80
    # So bitonic sort for 16 wires uses 80 comparators.
    
    def bitonic_merge(n, offset):
        if n == 2:
            return [(offset, offset + 1)]
        c1 = bitonic_merge(n // 2, offset)
        c2 = bitonic_merge(n // 2, offset + n // 2)
        c3 = [(offset + i, offset + n // 2 + i) for i in range(n // 2)]
        return c1 + c2 + c3
    
    def bitonic_sort(n, offset):
        if n == 2:
            return [(offset, offset + 1)]
        c1 = bitonic_sort(n // 2, offset)
        c2 = bitonic_sort(n // 2, offset + n // 2)
        c3 = bitonic_merge(n, offset)
        return c1 + c2 + c3
    
    return bitonic_sort(16, 0)
# EVOLVE-BLOCK-END

