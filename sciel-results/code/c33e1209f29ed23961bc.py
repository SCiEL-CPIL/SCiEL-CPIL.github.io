import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145
    
    if k != 3:
        return _luw_construct(k, g)
        
    random.seed(42)
    
    # Try Heawood Lift (n=518)
    result = _try_heawood_lift(g, deadline)
    if result:
        return result
        
    # Try Petersen Lift (n=520, 530...)
    result = _try_petersen_lift(g, deadline)
    if result:
        return result
        
    # Fallback
    # ...
