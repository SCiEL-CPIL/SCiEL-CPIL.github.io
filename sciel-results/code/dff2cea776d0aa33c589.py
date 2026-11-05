import numpy as np
import time

N = 400

def _value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _optimize(h0, deadline, rng):
    h = h0.astype(float).copy()
    s = np.sum(h)
    if abs(s) < 1e-12:
        h += 1e-6
        s = np.sum(h)
    # project to sum=1
    h = h - (s - 1.0)/N
    best_h = h.copy()
    best_val = _value(h)
    tau = 0.002
    lr = 0.01
    it = 0
    while time.time() < deadline:
        c = np.convolve(h, h)
        cm = c.max()
        e = np.exp((c - cm)/tau)
        p = e/e.sum()
        conv_part = np.convolve(p, h[::-1], 'full')[N-1:2*N-1]
        grad = (2.0/tau)*conv_part
        h = h - lr*grad
        h = h - (np.sum(h) - 1.0)/N
        if not np.all(np.isfinite(h)):
            h = best_h
            break
        it += 1
        if it % 50 == 0:
            tau *= 0.97
            lr *= 0.995
        if it % 10 == 0:
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 105.0
    rng = np.random.default_rng(2024)
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # structured inits
    inits = [np.ones(N)]
    i = np.arange(N)
    inits.append(1.0 + 2.0*np.cos(np.pi*i/(N-1)))          # U shape
    inits.append(1.0 - 2.0*np.cos(np.pi*i/(N-1)))          # inverted U
    inits.append(1.0 + 2.0*np.cos(np.pi*i/(N-1) + np.pi/2)) # shifted
    for _ in range(20):
        inits.append(1.0 + 0.3*rng.standard_normal(N))
    
    n = len(inits)
    for r, h0 in enumerate(inits):
        if time.time() > deadline:
            break
        slice_end = start + (deadline - start)*(r+1)/n
        h, v = _optimize(h0, slice_end, rng)
        if v < best_val:
            best_val = v
            best_h = h
    
    return best_h
