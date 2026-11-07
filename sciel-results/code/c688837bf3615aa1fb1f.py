import numpy as np
import time

_MODULE_START_TIME = time.time()
_DEADLINE = _MODULE_START_TIME + 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # ...
    if time.time() > _DEADLINE:
         # Return a safe fallback, e.g., mean of params or zeros
         # Since we don't have the target, returning zeros or a simple polynomial is best.
         # But we are inside the optimization. If we return garbage, the optimizer might fail.
         # However, if we are over deadline, the whole run is likely to be killed.
         # The directive says "return the best result found". We don't have a "best result" stored.
         # We just return the current prediction quickly.
         pass
