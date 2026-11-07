import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using bounded trigonometric basis functions.
    
    This approach avoids high-order polynomial instability by using sin/cos
    for non-linearities and includes interaction terms between state and time.
    
    Model:
    a = p0*sin(w*t) + p1*cos(w*t) + p2*sin(x) + p3*cos(x) + p4*sin(v) + p5*cos(v) 
      + p6*sin(x + w*t) + p7*cos(x - w*t) + p8*(x*v) + p9*(x*v*sin(w*t))
      
    Note: We need to fit the frequency w as well. That's 10 params total.
    Let's map:
    p0: freq w
    p1: amp sin(wt)
    p2: amp cos(wt)
    p3: amp sin(x)
    p4: amp cos(x)
    p5: amp sin(v)
    p6: amp cos(v)
    p7: amp sin(x+wt)
    p8: amp cos(x-wt)
    p9: amp x*v term? 
    Wait, I have 10 slots.
    
    Let's assign:
    w = params[0]
    a1 = params[1] # sin(wt)
    a2 = params[2] # cos(wt)
    b1 = params[3] # sin(x)
    b2 = params[4] # cos(x)
    c1 = params[5] # sin(v)
    c2 = params[6] # cos(v)
    d1 = params[7] # sin(x + wt)
    d2 = params[8] # cos(x - wt)
    e1 = params[9] # x*v
    
    This covers time, state, and interactions.
    """
    w = params[0]
    a1 = params[1]
    a2 = params[2]
    b1 = params[3]
    b2 = params[4]
    c1 = params[5]
    c2 = params[6]
    d1 = params[7]
    d2 = params[8]
    e1 = params[9]
    
    wt = w * t
    sin_wt = np.sin(wt)
    cos_wt = np.cos(wt)
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    sin_x_wt = np.sin(x + wt)
    cos_x_wt = np.cos(x - wt)
    
    xv = x * v
    
    result = (
        a1 * sin_wt + 
        a2 * cos_wt + 
        b1 * sin_x + 
        b2 * cos_x + 
        c1 * sin_v + 
        c2 * cos_v + 
        d1 * sin_x_wt + 
        d2 * cos_x_wt + 
        e1 * xv
    )
    
    return result
