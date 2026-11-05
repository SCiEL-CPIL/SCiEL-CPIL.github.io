import numpy as np
import time

N = 11
T0 = time.time()
DEADLINE = T0 + 145.0

def _hv(cx, cy, R, a):
    t = a + np.arange(6) * (np.pi/3)
    return np.column_stack([cx + R*np.cos(t), cy + R*np.sin(t)])

def _on(a):
    t = a + np.pi/6 + np.arange(6) * (np.pi/3)
    return np.column_stack([np.cos(t), np.sin(t)])

def _sep(v1, v2):
    """Return >0 if overlapping (min sep), <0 if separated."""
    b = np.inf
    for k in range(6):
        e = v1[(k+1)%6] - v1[k]
        n = np.array([-e[1], e[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-12: continue
        n /= nn
        p1 = v1 @ n; p2 = v2 @ n
        s = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if s < 0: return s
        if s < b: b = s
        e2 = v2[(k+1)%6] - v2[k]
        n2 = np.array([-e2[1], e2[0]])
        nn2 = np.linalg.norm(n2)
        if nn2 < 1e-12: continue
        n2 /= nn2
        p1b = v1 @ n2; p2b = v2 @ n2
        s2 = min(p1b.max(), p2b.max()) - max(p1b.min(), p2b.min())
        if s2 < 0: return s2
        if s2 < b: b = s2
    return b

def _cost(pos, ang, oc, oa):
    nv = np.array([_hv(pos[i,0], pos[i,1], 1.0, ang[i]) for i in range(N)])
    on = _on(oa)
    Lm = 0.0
    for i in range(N):
        d = nv[i] - oc
        Lm = max(Lm, (d @ on.T).max() * 2.0 / np.sqrt(3.0))
    op = 0.0
    for i in range(N):
        for j in range(i+1, N):
            s = _sep(nv[i], nv[j])
            if s > 0:
                op += s * s
    return Lm, op

def construct():
    s3 = np.sqrt(3.0)
    sp = 1.75
    d1 = np.array([sp, 0.0])
    d2 = np.array([sp*0.5, sp*s3/2])
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pts.append(i*d1 + j*d2)
    pos = np.array(pts[:N])
    ang = np.zeros(N)
    oc = np.array([0.0, 0.0])
    oa = 0.0
    L, op = _cost(pos, ang, oc, oa)
    bL = L; bP = pos.copy(); bA = ang.copy()
    T = 2.0; st = 0.4
    while time.time() < DEADLINE:
        i = np.random.randint(N)
        dp = np.random.normal(0, st, 2)
        da = np.random.normal(0, st*0.3)
        p2 = pos.copy(); a2 = ang.copy()
        p2[i] += dp; a2[i] = (a2[i] + da) % (2*np.pi)
        L2, op2 = _cost(p2, a2, oc, oa)
        c2 = L2 + 20*op2; c1 = L + 20*op
        if c2 < c1 or np.random.random() < np.exp(-(c2-c1)/max(T,1e-6)):
            pos, ang, L, op = p2, a2, L2, op2
            if op < 1e-8 and L < bL:
                bL = L; bP = pos.copy(); bA = ang.copy()
        T *= 0.997
        if T < 0.05: T = 0.05; st *= 0.95
        if st < 0.01: break
    inner = [[float(bP[i,0]), float(bP[i,1]), float(np.degrees(bA[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(bL), 0.0
