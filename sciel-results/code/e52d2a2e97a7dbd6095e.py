import time

def construct():
    start_time = time.time()
    deadline = 150.0
    # ... construction ...
    if time.time() > deadline - 5.0:
         # return best found, but here it's deterministic
         pass
    return {"U": U, "V": V, "W": W}
