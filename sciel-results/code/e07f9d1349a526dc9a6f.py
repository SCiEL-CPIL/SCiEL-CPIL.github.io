import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Implement the Lazebnik-Woldar construction for q=4
    # n = 16, colors = 5
    n = 16
    colors = [0] * (n * (n - 1) // 2)
    
    # Finite Field F_4 arithmetic
    # Elements: 0, 1, 2 (omega), 3 (omega+1)
    # omega^2 = omega + 1 => 2^2 = 2 + 1 = 3
    # (omega+1)^2 = omega^2 + 2omega + 1 = (omega+1) + 2omega + 1 = 3omega + 2 = 3(omega) + 2
    # 3omega = 3(omega) = (omega+1)omega = omega^2 + omega = (omega+1) + omega = 2omega + 1 = 2(2) + 1 = 5 = 1 (mod 2? No, field of 4)
    # Let's use standard representation:
    # 0: 0
    # 1: 1
    # 2: x
    # 3: x+1
    # x^2 = x+1
    # x^3 = x(x+1) = x^2+x = (x+1)+x = 1
    # x^4 = x
    # Inverse:
    # inv(0) undefined
    # inv(1) = 1
    # inv(2) = 2 (since 2*2 = 4 = 1 in F4? 2*2 = x^2 = x+1 = 3. No.
    # 2*3 = x(x+1) = x^2+x = 1. So inv(2)=3, inv(3)=2.
    
    inv = {0: 0, 1: 1, 2: 3, 3: 2}
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Map i, j to (x, y) in F_4 x F_4
            xi = i // 4
            yi = i % 4
            xj = j // 4
            yj = j % 4
            
            dx = (xj - xi) % 4
            dy = (yj - yi) % 4
            
            if dx == 0:
                c = 0
            else:
                # m = dy / dx
                # Note: dy, dx are in 0..3 representing F4
                m = (dy * inv[dx]) % 4
                # Map m (0,1,2,3) to colors 1,2,3,4
                c = m + 1
            
            colors[idx] = c
            idx += 1
            
            if time.time() > deadline:
                # Return current best if timeout (shouldn't happen for n=16)
                return {'n': n, 'colors': colors}

    return {'n': n, 'colors': colors}
