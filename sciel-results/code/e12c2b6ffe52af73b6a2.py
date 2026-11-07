"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    
    # Partition vertices into two disjoint sets of size 'half'
    set_a = tuple(range(half))
    set_b = tuple(range(half, n))
    
    # Generate all edges between set_a and set_b using itertools for structural distinction
    raw_pairs = itertools.product(set_a, set_b)
    
    # Convert generator to a list of lists, ensuring types are correct
    # Using a list comprehension with explicit type casting to int
    edges = [[int(u), int(v)] for u, v in raw_pairs]
    
    return edges
# EVOLVE-BLOCK-END
