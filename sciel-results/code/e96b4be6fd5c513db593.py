   while bad_count > 0 and time.time() < deadline:
       # Find bad cycle
       # Try to fix
       # If fixed, update bad_count (decrement by 1, but check if others broke? 
       # We checked others, so they are good. So bad_count -= 1)
       # If not fixed, remove edge, recalculate bad_count from scratch (expensive but rare?)
   