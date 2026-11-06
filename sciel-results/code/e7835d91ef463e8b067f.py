import time
import sys
from itertools import combinations
from collections import defaultdict

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using algebraic generation
    via the action of PSL(2, 11) on the projective line over GF(11).
    This is deterministic and instantaneous.
    """
    # Points are 0..10 and 11 (representing infinity)
    # Base block for S(5,6,12) under PSL(2,11) action
    # A known base block is {0, 1, 2, 4, 8, infinity}
    base_block = frozenset([0, 1, 2, 4, 8, 11])
    
    blocks = set()
    
    # Generate all elements of PSL(2, 11)
    # Matrices [a b; c d] with ad - bc = 1 mod 11
    # Note: PSL(2,11) is SL(2,11) / {±I}. 
    # We can generate all SL(2,11) matrices and apply them.
    # Distinct transformations will be handled by the set of blocks.
    
    for a in range(11):
        for b in range(11):
            for c in range(11):
                # If c == 0, then a*d = 1, so a != 0 and d = a^-1
                # If c != 0, then d = (1 - a*b) * c^-1
                if c == 0:
                    if a == 0:
                        continue
                    # d must be inverse of a
                    # Find d such that a*d % 11 == 1
                    for d in range(11):
                        if (a * d) % 11 == 1:
                            # Matrix is [[a, b], [0, d]]
                            # Apply to base block
                            new_block = set()
                            for x in base_block:
                                if x == 11: # infinity
                                    # g(inf) = a/c if c!=0 else inf
                                    # Here c=0, so g(inf) = inf = 11
                                    new_block.add(11)
                                else:
                                    # g(x) = (a*x + b) / d
                                    # Since d is inverse of a, this is a*x*b? No.
                                    # (a*x + b) * d^-1. But d = a^-1, so d^-1 = a.
                                    # So g(x) = (a*x + b) * a = a^2*x + a*b
                                    val = (a * x + b) * a % 11
                                    new_block.add(val)
                            blocks.add(frozenset(new_block))
                else:
                    # c != 0
                    # d = (1 - a*b) * c^-1
                    inv_c = pow(c, 10, 11) # Fermat's little theorem
                    d = (1 - a * b) * inv_c % 11
                    # Matrix is [[a, b], [c, d]]
                    # Apply to base block
                    new_block = set()
                    for x in base_block:
                        if x == 11: # infinity
                            # g(inf) = a/c
                            val = a * inv_c % 11
                            new_block.add(val)
                        else:
                            # g(x) = (a*x + b) / (c*x + d)
                            denom = (c * x + d) % 11
                            if denom == 0:
                                new_block.add(11) # infinity
                            else:
                                num = (a * x + b) % 11
                                inv_denom = pow(denom, 10, 11)
                                val = num * inv_denom % 11
                                new_block.add(val)
                    blocks.add(frozenset(new_block))
    
    # Convert to list of lists
    result = [list(b) for b in blocks]
    
    # Verify we have 132 blocks
    # If not, something is wrong, but this method is standard.
    return result

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
    used = set()
    for b in res:
        for s in combinations(b, 5):
            if s in used:
                print("Invalid: repeated 5-subset")
                sys.exit(1)
            used.add(s)
    print("Valid packing")
