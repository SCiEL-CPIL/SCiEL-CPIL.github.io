import time
_start_time = time.time()
_TIMEOUT = 320.0 # Safety margin

def priority(vector, n):
    # Check timeout
    if time.time() - _start_time > _TIMEOUT:
        return -1e9 # Return very low priority to stop adding? Or just return fast.
    
    c0 = 0
    c1 = 0
    c2 = 0
    h = 0
    
    # Single pass
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        h = (h * 31 + v) % 1000000007
            
    ideal = n / 3.0
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # We want to maximize priority.
    # Minimize dev.
    # Tie break with hash.
    # If dev is same, prefer higher hash? Or lower?
    # Deterministic order is good.
    
    return -dev - h * 1e-9
