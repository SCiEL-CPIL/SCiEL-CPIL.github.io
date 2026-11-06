"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    roots = {}
    for x in range(1, 1009):
        r = (x * x) % 1009
        if r not in roots:
            roots[r] = x
            
    # Ascending search finds the minimum s
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
            
    raise RuntimeError('No valid run found within deadline')
# EVOLVE-BLOCK-END
