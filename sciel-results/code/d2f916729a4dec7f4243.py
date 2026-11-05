"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: evaluator kills after 150s, use 145s as hard stop
    start_time = time.time()
    deadline = start_time + 145.0
    
    # For OneMax, the optimal solution is all 1s.
    # This is a direct construction task, not an evolutionary search.
    # The optimal answer is simply "1" * N.
    # We return this immediately since it's provably optimal.
    return "1" * N
# EVOLVE-BLOCK-END
