    if time.time() > deadline:
        # Fallback or just return current state (which is complete)
        pass
        
    return {"U": U, "V": V, "W": W}
  