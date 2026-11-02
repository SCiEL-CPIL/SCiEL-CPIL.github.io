import time
import numpy as np

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # LUW parameters
    q = k
    # d such that girth >= g
    # Girth of CD(d,q) is d+5 (odd d) or d+4 (even d)
    # We need girth >= 16
    # If d=11 (odd), girth >= 16.
    d = 11 if g <= 16 else (g - 5 if (g - 5) % 2 else g - 4)
    
    # For q=3, d=11, N = 2 * 3^(12) is too big for memory/time in Python lists.
    # But we can return the structure.
    # However, the evaluator might not handle 1M vertices well.
    # But it's the only valid explicit construction.
    
    # If q=3, d=11, N is huge.
    # Is there a smaller d?
    # If d=10, girth 14. Not enough.
    
    # Let's check if we can use a smaller q?
    # k=3, so q=3 is fixed.
    
    # We must return LUW.
    
    # Optimized LUW construction
    # ...
