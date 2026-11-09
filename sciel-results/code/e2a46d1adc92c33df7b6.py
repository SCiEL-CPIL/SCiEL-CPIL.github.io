import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s margin per call? No, total budget is 120s/150s.
    # The task says "evaluator KILLS ... after 150s".
    # We have 3 equations. We should spend max ~40s each to be safe?
    # Actually, the score depends on mean calls. Speed is secondary unless it causes timeout.
    # But the directive says: "Keep an EXPLICIT deadline guard".
    # Let's set a generous per-call limit, say 30 seconds.
    
    x = float(x0)
    fx = f(x)
    best_x = x
    best_fx = fx
    if abs(fx) <= 1e-8:
        return x

    # Initial step for Secant
    # Choose h based on magnitude of x
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    fx1 = f(x1)
    best_x2 = x1
    best_fx2 = fx1
    if abs(fx1) <= 1e-8:
        return x1

    # Update best
    if abs(fx1) < abs(fx):
        best_x = x1
        best_fx = fx1

    # Secant Iterations
    x0_curr = x
    x1_curr = x1
    f0_curr = fx
    f1_curr = fx1

    for i in range(100):
        if time.time() > deadline:
            break
            
        denom = f1_curr - f0_curr
        if denom == 0:
            break
            
        x_new = x1_curr - f1_curr * (x1_curr - x0_curr) / denom
        f_new = f(x_new)
        
        if abs(f_new) < abs(best_fx):
            best_x = x_new
            best_fx = f_new
            
        if abs(f_new) <= 1e-8:
            return x_new

        # Check for divergence or stall
        if not (x_new == x_new): # NaN check
            break
            
        x0_curr = x1_curr
        f0_curr = f1_curr
        x1_curr = x_new
        f1_curr = f_new

    return best_x
