"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all '1's.
    Since the objective is deterministic and known (maximize count of '1's),
    we directly return the global optimum.
    """
    return "1" * N
# EVOLVE-BLOCK-END
