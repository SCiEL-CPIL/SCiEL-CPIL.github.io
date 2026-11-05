"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import random

def construct():
    all_sets = list(combinations(range(1, 17), 4))
    all_frozen = [frozenset(s) for s in all_sets]
    
    chosen = []
    
    for s in all_frozen:
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                a = chosen[i]
                b = chosen[j]
                ab = a & b
                as_ = a & s
                bs = b & s
                if ab == as_ == bs:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(s)
    
    return [sorted(s) for s in chosen]
# EVOLVE-BLOCK-END
