    T = 1.0
    cooling_rate = 0.999
    min_T = 0.05
    
    while time.time() < deadline:
        # ... SA step ...
        
        T *= cooling_rate
        if T < min_T:
            T = 1.0 # Reheat
