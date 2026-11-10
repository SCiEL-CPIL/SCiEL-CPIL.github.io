# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2.
    
    Strategy: Use a Hermitian unital-based construction to create a set S
    (the excluded points) such that N = F_{49}^2 \\ S is a Nikodym set.
    The goal is to maximize |S| to maximize the complement size 2401 - |N| = |S|.
    
    For F_q^2 where q = p^2, a known construction uses the Hermitian curve.
    The unital U = {(x, y) : Tr(x) + Norm(y) = 0} has q+1 points.
    We can extend S beyond the unital by adding points while preserving the Nikodym property.
    
    Key insight: If S is a set such that for every point x in S and every direction v,
    the line {x + t*v : t in F_49} contains at most one point of S (namely x itself),
    then N = F_49^2 \\ S is a Nikodym set.
    
    Actually, the condition is: for every point x in F_49^2 (not just in S), there exists
    a line through x (with direction (1,m) or (0,1)) such that all other points on that line are in N.
    This is equivalent to: for every x and every direction v, if x is not in N (i.e., x in S),
    then the line through x with direction v must have all other points in N.
    Wait, no. The condition is: for every x in F_49^2, there EXISTS a direction v such that
    all points on the line through x with direction v, except x itself, are in N.
    
    If x is in N, this is trivially satisfied by any direction (since x itself is in N, and we need
    the OTHER points to be in N). Wait, no: "whose points other than x all belong to N". So if x is in N,
    we need to find a line through x where all other points are in N. If x is in S, we need to find
    a line through x where all other points are in N (i.e., no other points of S on that line).
    
    So the condition is: for every x in F_49^2, there exists a direction v such that the line
    {x + t*v : t in F_49} \\ {x} is a subset of N. Equivalently, the line through x with direction v
    contains no other points of S besides possibly x.
    
    If x is in S: we need a line through x with no other points of S.
    If x is in N: we need a line through x with no points of S (since x itself is not in S, all points
    on the line except x must be in N, meaning no points of S on the line).
    
    Wait, if x is in N, the condition is: all points other than x on the line are in N. Since x is in N,
    this means the entire line is in N, i.e., no points of S on the line.
    
    So: for every x in F_49^2, there exists a direction v such that the line through x with direction v
    contains no points of S (if x in N) or contains only x from S (if x in S).
    
    In both cases: the line through x with direction v contains at most one point of S (which is x if x in S,
    or zero points if x in N).
    
    This is a strong condition. Let's use the unital construction.
    
    For F_{49}^2, p=7, q=49. The unital U has q+1 = 50 points.
    A known result: if we take S to be a set of points such that no line contains more than one point of S,
    then N = F_{49}^2 \\ S is a Nikodym set. This is because for any x, we can find a line through x
    that avoids S (since there are q=49 lines through x, and S has at most q+1 points, by pigeonhole
    or counting arguments).
    
    Actually, the maximum size of a set S with no three collinear points in F_q^2 is related to caps.
    But we need a stronger condition: no line contains more than one point of S.
    
    The maximum size of such a set is q (a "regulus" or similar structure). For example, the graph of a
    function f: F_q -> F_q has the property that no vertical line contains more than one point, but
    other lines might contain multiple points.
    
    Let's try a different approach: use the fact that for the unital U, any line intersects U in either
    1 or 3 points (for F_{p^2}^2). So U itself doesn't satisfy "no line contains more than one point of S".
    
    Let me reconsider. The seed program uses S = {points in U with ya in QR}. Let's compute |S| for the
    current approach and see if we can do better.
    
    Actually, let me try a simpler construction: take S to be a line minus one point. Then |S| = 48.
    But the current program has |S| = 196 (approximately), which is much larger.
    
    Let me just implement the current approach but verify it's correct, and try to extend S.
    """
    p = 7
    q = p * p  # 49
    
    # Find smallest quadratic non-residue mod 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3
    
    # Quadratic residues mod p
    qr = set((u * u) % p for u in range(p))
    # qr = {0, 1, 2, 4}
    
    # Check if point is in unital U: Tr(x) + Norm(y) = 0
    # Tr(xa + xb*alpha) = 2*xa
    # Norm(ya + yb*alpha) = ya^2 - k*yb^2
    def in_unital(xa, xb, ya, yb):
        return (2 * xa + ya * ya - k * yb * yb) % p == 0
    
    # Build S: points in U with ya in qr
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if in_unital(xa, xb, ya, yb) and ya in qr:
                        S.add((xa, xb, ya, yb))
    
    # N = F_{49}^2 \\ S
    N_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in S:
                        N_points.append((xa, xb, ya, yb))
    
    # Convert to (k, 2, 2) array
    result = np.zeros((len(N_points), 2, 2), dtype=np.int64)
    for i, (xa, xb, ya, yb) in enumerate(N_points):
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    return result

# EVOLVE-BLOCK-END
