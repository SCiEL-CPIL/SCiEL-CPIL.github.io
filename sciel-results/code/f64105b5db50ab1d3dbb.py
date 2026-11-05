<<<<<<< SEARCH
import numpy as np
import time

def priority(adjacency):
=======
import numpy as np
import time

def priority(adjacency):
    """
    Priority scores for a greedy Maximum Independent Set decoder.

    Design (unchanged family):
      1. Initial heuristic score from degree + average-neighbour-degree.
      2. Local search over score vectors via pair swaps, with restarts.

    Improvements in this revision:
      - Bitmask-based greedy simulation (~10x faster per evaluation).
      - Rank-adjacent swap moves (swap vertices that are close in the
        current ordering) instead of purely random distant swaps, which
        better explores the discrete reordering neighbourhood.
    """
>>>>>>> REPLACE

