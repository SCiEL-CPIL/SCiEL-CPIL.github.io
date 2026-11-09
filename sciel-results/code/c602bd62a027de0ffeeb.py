"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the Pawlik (Černý) automaton for N=12.
    
    This automaton is known to achieve the theoretical maximum reset threshold
    of (N-1)^2 = 121 for N states.
    
    Transition definitions:
    - Letter 0 (sigma0): cyclic permutation, i -> (i + 1) % N
    - Letter 1 (sigma1): 
        - state 0 maps to 0 (fixed point)
        - state i (i > 0) maps to i - 1
    
    The structure creates a "long chain" where synchronization requires
    carefully navigating the cycle with letter 0 to align states before
    applying letter 1 to collapse them, leading to the maximum threshold.
    """
    # Letter 0: cyclic shift
    sigma0 = [(i + 1) % N for i in range(N)]
    
    # Letter 1: fix 0, shift others down by 1
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    
    return [sigma0, sigma1]
# EVOLVE-BLOCK-END
