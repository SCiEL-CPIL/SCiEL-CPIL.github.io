# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges of K_{7,8} on vertices 0..14.
    Part A = {0..6}, Part B = {7..14}.
    All A vertices have degree 8, all B vertices have degree 7.
    Any 3-edge path s-u-v-t has endpoints s,t that are neighbors of u and v respectively.
    In K_{7,8}, neighbors of any A vertex are all B vertices, and neighbors of any B vertex are all A vertices.
    So s and t are in opposite parts from u and v.
    
    Case analysis for path s-u-v-t:
    - If u in A, v in B: s in B, t in A. deg(s)=7, deg(t)=8. Not equal.
    - If u in B, v in A: s in A, t in B. deg(s)=8, deg(t)=7. Not equal.
    - s and t cannot both be in A (since u would need to be in B and v in B, but no B-B edges).
    - s and t cannot both be in B (since u would need to be in A and v in A, but no A-A edges).
    
    So no 3-edge path has equal-degree endpoints. Edge count = 56.
    """
    n = 15
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    # Ensure exact types: Python int for vertices, list of lists
    result = [[int(u), int(v)] for u, v in edges]
    return result
# EVOLVE-BLOCK-END
