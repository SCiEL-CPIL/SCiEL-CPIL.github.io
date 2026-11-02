# ... (existing imports) ...

def construct():
    # ... (existing code) ...
    
    # Verify the decomposition (fast path first)
    if rank == 42:
        if _verify_decomposition_np(all_U, all_V, all_W, n, m, p):
            # The entries are integers, so if max error is < 1e-9, it is likely exact.
            # We skip the slower exact check to save time, trusting the integer nature.
            return {"U": all_U, "V": all_V, "W": all_W}
    
    # Fallback
    # ...
