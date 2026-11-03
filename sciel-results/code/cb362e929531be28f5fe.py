<<<<<<< SEARCH
def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin for 150s limit
    
    # Phase 1: Fibonacci spiral initialization
    x = _fibonacci_spiral(N)
    
    # Phase 2: Momentum optimization
    phase1_limit = start_time + 60.0
    x, best_energy = _momentum_optimize(x, start_time, phase1_limit)
    
    # Phase 3: L-BFGS polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 10:
        x = _lbfgs_polish(x, remaining - 5)
    
    # Final validation
    x = _project_to_sphere(x)
    return x
=======
def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin for 150s limit
    
    # Phase 1: Fibonacci spiral initialization
    x = _fibonacci_spiral(N)
    
    # Phase 2: Momentum optimization
    phase1_limit = start_time + 50.0
    x, best_energy = _momentum_optimize(x, start_time, phase1_limit)
    
    # Phase 3: L-BFGS polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 20:
        x = _lbfgs_polish(x, remaining - 15)
    
    # Phase 4: Final high-precision gradient polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 5:
        x = _final_grad_polish(x, remaining - 3)
    
    # Final validation
    x = _project_to_sphere(x)
    return x
>>>>>>> REPLACE
