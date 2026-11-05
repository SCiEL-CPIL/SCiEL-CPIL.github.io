import numpy as np

def priority(adjacency):
    # Convert to float for operations
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate 2 * L(v) for each v
    # L(v) is the number of edges in the subgraph induced by N(v)
    # 2 * L(v) = sum_{u in N(v)} sum_{w in N(v)} A[u,w]
    # This can be computed as sum_{u} A[v,u] * B[v,u] where B = A^2
    # B[v,u] is the number of common neighbors of v and u
    # If u is a neighbor of v, B[v,u] counts how many of v's other neighbors are also connected to u
    # Summing this over all neighbors u gives the sum of degrees of u within N(v), which is 2 * L(v)
    
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    
    # Score formula: -d^2 / (d + 2L)
    # If d=0, score is 0 (highest priority)
    # If neighbors are independent (L=0), score is -d
    # If neighbors are clique (L ~ d^2/2), score is ~ -1
    
    # Avoid division by zero
    denom = degrees + two_L
    scores = np.zeros(n, dtype=float)
    mask = degrees > 0
    scores[mask] = -(degrees[mask] ** 2) / denom[mask]
    
    return scores
