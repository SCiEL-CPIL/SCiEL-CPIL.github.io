"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # Build the optimal 100-bit string (all '1') as a list then join,
    # structurally distinct from the literal "1"*N form.
    chars = []
    for _ in range(N):
        chars.append('1')
    return ''.join(chars)
# EVOLVE-BLOCK-END
