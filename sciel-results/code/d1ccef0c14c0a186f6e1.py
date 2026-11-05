def line_intersection(p1,p2,p3,p4):
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    d=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if d==0: return None
    px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/d
    py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/d
    if abs(px-round(px))>1e-9 or abs(py-round(py))>1e-9: return None
    px=int(round(px)); py=int(round(py))
    if not(-BOUND<=px<=BOUND and -BOUND<=py<=BOUND): return None
    return (px,py)
