P_not_N = P.difference(adj[pivot]) if pivot is not None else P
for v in list(P_not_N):
    ...
    P.remove(v)
    X.add(v)
