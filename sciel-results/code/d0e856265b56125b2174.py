import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    roots = {}
    for x in range(1, 1009):
        if time.time() > deadline:
            # Fallback if somehow slow (unlikely)
            break
        r = (x * x) % 1009
        if r != 0 and r not in roots:
            roots[r] = x
            
    best_result = None
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if all((s + i) in roots for i in range(5)):
            best_result = {'start': s, 'roots': [roots[s + i] for i in range(5)]}
            break # Found minimum s, no need to continue
            
    if best_result:
        return best_result
        
    # If no result found (should not happen), return a valid one if possible or raise
    # Since s=1 is valid, this block is unreachable in practice.
    raise RuntimeError('No baseline run found')
