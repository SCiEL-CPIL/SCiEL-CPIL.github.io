import time

def construct():
    p = 1009
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    # Precompute quadratic residues and one root for each
    # roots[r] = x such that x*x % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
            
    # We want the minimum s.
    # s ranges from 1 to 1004.
    # Check s, s+1, s+2, s+3, s+4.
    # All must be in roots.
    
    # Since p is small (1009), a simple loop is sufficient and will finish in milliseconds.
    # No need for complex search or deadline guards really, but we keep it for safety.
    
    for s in range(1, 1005):
        if time.time() > deadline:
             # Fallback: return a known valid solution if we ran out of time
             # But this loop is so fast it won't happen.
             # We can just break and return best found or raise.
             # For safety, we'll just return the first one found in a secondary check or raise.
             pass
        
        valid = True
        rs = []
        for i in range(5):
            val = s + i
            if val in roots:
                rs.append(roots[val])
            else:
                valid = False
                break
        
        if valid:
            return {'start': s, 'roots': rs}
            
    # If no s found (should not happen), raise error
    raise RuntimeError("No valid start found")
