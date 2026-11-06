import time
import random

def _score_fast(points):
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ac = {}
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i+1, n):
            dx1 = xs[j] - xi
            dy1 = ys[j] - yi
            for k in range(j+1, n):
                a = dx1*(ys[k]-yi) - (xs[k]-xi)*dy1
                if a < 0:
                    a = -a
                if a > 0:
                    ac[a] = ac.get(a, 0) + 1
    if not ac:
        return 0
    return max(ac.values())

def construct():
    t0 = time.time()
    deadline = t0 + 75
    
    # 3 lines of 10
    best = [(x, y) for y in range(3) for x in range(10)]
    best_s = _score_fast(best)
    
    # Try shifts and scales of 3-line structure
    for ys in [1, 2, 3]:
        for xs in [1, 2, 3]:
            for yo in range(-15, 16):
                for xo in range(-15, 16):
                    if time.time() > deadline:
                        return best
                    pts = [(xo + x*xs, yo + y*ys) for y in range(3) for x in range(10)]
                    if all(-30 <= a <= 30 and -30 <= b <= 30 for a, b in pts):
                        s = _score_fast(pts)
                        if s > best_s:
                            best_s = s
                            best = pts[:]
    
    # Local search
    random.seed(7)
    cur = best[:]
    cur_s = best_s
    stall = 0
    while time.time() < deadline and stall < 30:
        imp = False
        i = random.randrange(30)
        ox, oy = cur[i]
        for dx in (-2,-1,0,1,2):
            for dy in (-2,-1,0,1,2):
                if dx==0 and dy==0: continue
                nx, ny = ox+dx, oy+dy
                if not (-30<=nx<=30 and -30<=ny<=30): continue
                np_ = cur[:]
                np_[i] = (nx,ny)
                if len(set(np_))<30: continue
                s = _score_fast(np_)
                if s > cur_s:
                    cur = np_
                    cur_s = s
                    imp = True
                    break
            if imp: break
        if imp:
            stall = 0
            if cur_s > best_s:
                best_s = cur_s
                best = cur[:]
        else:
            stall += 1
            if stall % 5 == 0:
                i2 = random.randrange(30)
                nx = random.randint(-30,30)
                ny = random.randint(-30,30)
                np_ = cur[:]
                np_[i2] = (nx,ny)
                if len(set(np_))==30:
                    s = _score_fast(np_)
                    if s >= cur_s:
                        cur = np_
                        cur_s = s
    return best
