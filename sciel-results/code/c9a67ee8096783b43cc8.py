def coefficients(stages):
    """
    Implements a 6-step Strang splitting using 12 stages.
    Each Strang step B(dt/2) A(dt) B(dt/2) is mapped to two stages:
    Stage 1: B(dt/2) A(dt)
    Stage 2: B(dt/2) A(0)
    This results in a symmetric 2nd order approximation.
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12, though task implies 12
        # We assume stages=12 based on task description "twelve-stage"
        pass
        
    a = [0.0] * stages
    b = [0.0] * stages
    
    num_strang_steps = stages // 2
    dt = 1.0 / num_strang_steps
    
    for j in range(num_strang_steps):
        idx1 = 2 * j
        idx2 = 2 * j + 1
        
        # Stage 1: B(dt/2) A(dt)
        b[idx1] = dt / 2.0
        a[idx1] = dt
        
        # Stage 2: B(dt/2) A(0)
        b[idx2] = dt / 2.0
        a[idx2] = 0.0
        
    return a, b
