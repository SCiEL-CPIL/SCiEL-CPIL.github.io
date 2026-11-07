_start_time = None

def quote(t, inventory, mid, params):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # If we are running out of time, return a safe fallback
    # 300s limit. Let's say if we've been running for 290s, we switch to fallback.
    # But since each call is fast, this is unlikely.
    # We'll just proceed.
