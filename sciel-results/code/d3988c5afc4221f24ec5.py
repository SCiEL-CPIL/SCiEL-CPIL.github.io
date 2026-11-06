import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def get_edge_idx(i, j, n):
        if i > j: i, j = j, i
        return i * (2 * n - i - 3) // 2 + j - i - 1

    def is_valid_full(colors, n):
        # Full check
        # Precompute partitions
        # This is slow, so only call if sampling says valid
        for combo in itertools.combinations(range(n), 6):
            if has_mono_k222(combo, colors, n):
                return False
        return True

    def has_mono_k222(combo, colors, n):
        # 15 partitions of 6 items
        # We check if any partition has all 12 edges same color
        # Generate partitions on the fly or hardcode?
        # Hardcoding is faster.
        # Partitions of {0,1,2,3,4,5}
        # P1: (0,1)(2,3)(4,5) -> Edges: 01,02,03,04,05,12,13,14,15,24,25,34,35,45? No.
        # Edges between pairs:
        # Pair A={0,1}, B={2,3}, C={4,5}
        # Edges: (0,2),(0,3),(1,2),(1,3) [A-B]
        # (0,4),(0,5),(1,4),(1,5) [A-C]
        # (2,4),(2,5),(3,4),(3,5) [B-C]
        
        # We need the colors of these 12 edges.
        # Indices in the 15-edge list of the 6-set:
        # 0: (0,1)
        # 1: (0,2)
        # 2: (0,3)
        # 3: (0,4)
        # 4: (0,5)
        # 5: (1,2)
        # 6: (1,3)
        # 7: (1,4)
        # 8: (1,5)
        # 9: (2,3)
        # 10: (2,4)
        # 11: (2,5)
        # 12: (3,4)
        # 13: (3,5)
        # 14: (4,5)
        
        # For P1: A={0,1}, B={2,3}, C={4,5}
        # Edges: 1,2,5,6 (A-B), 3,4,7,8 (A-C), 10,11,12,13 (B-C)
        # Check if all 12 are 0 or all 12 are 1.
        
        # We can precompute the list of 12 indices for each of the 15 partitions.
        pass

    # ... implementation ...
