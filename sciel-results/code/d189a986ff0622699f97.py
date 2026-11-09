<<<<<<< SEARCH
"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Start with the 5 disjoint K4s (30 edges) as a valid baseline.
- Use bit-parallel representation for fast validity checks.
- Perform a multi-restart local search with:
    * Greedy edge addition
    * Simulated annealing with swap moves (remove one edge, add another)
    * Periodic full K2,3 validity re-checks
- Return the best graph found within the time budget.
"""

import time
import random
import math
=======
"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Start with the 5 disjoint K4s (30 edges) as a valid baseline.
- Use NumPy vectorized adjacency for fast parallel common-neighbor checks.
- Perform a multi-restart local search with:
    * Vectorized greedy edge addition (parallel validity checks)
    * Simulated annealing with swap moves
    * Systematic final polish pass with all-pairs checks
- Return the best graph found within the time budget.
"""

import time
import random
import math
import numpy as np
>>>>>>> REPLACE

