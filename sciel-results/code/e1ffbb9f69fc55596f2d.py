def _merge(left, right):
    a, b = len(left), len(right)
    if a == 0 or b == 0:
        return []
    if a == 1 and b == 1:
        i, j = left[0], right[0]
        return [(min(i,j), max(i,j))]
    
    comps = []
    le, lo = left[0::2], left[1::2]
    re, ro = right[0::2], right[1::2]
    
    comps.extend(_merge(le, re))
    comps.extend(_merge(lo, ro))
    
    W1 = le + re
    W2 = lo + ro
    
    for i in range(min(len(W1), len(W2))):
        x, y = W1[i], W2[i]
        comps.append((min(x,y), max(x,y)))
        if i + 1 < len(W1):
            x, y = W2[i], W1[i+1]
            comps.append((min(x,y), max(x,y)))
    
    if len(W1) > len(W2):
        x, y = W2[-1], W1[len(W2)]
        comps.append((min(x,y), max(x,y)))
    elif len(W2) > len(W1):
        x, y = W1[-1], W2[len(W1)]
        comps.append((min(x,y), max(x,y)))
    
    return comps
