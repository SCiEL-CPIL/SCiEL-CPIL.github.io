def sdp_warmstart(time_budget):
    # Shor relaxation, alternating projections
    # X is (N+1)x(N+1)
    n=N+1
    # init X
    X=np.eye(n)
    # set first row/col to a feasible h (e.g., 0.5)
    h0=np.full(N,0.5)
    X[0,1:]=h0; X[1:,0]=h0
    # make X PSD initially: X = something PSD. Use X = y y^T with y=[1,h0]
    y=np.concatenate([[1.0],h0])
    X=np.outer(y,y)
    t=25.0
    for it in range(2000):
        if time.time()-START>time_budget: break
        # project onto affine+box+halfspace (in X,t)
        # 1) X[0,0]=1
        X[0,0]=1.0
        # 2) first row/col: adjust to sum=N/2, 0<=..<=1, symmetric
        hr=X[0,1:].copy()
        hr=project(hr)  # sum=N/2, in[0,1]
        X[0,1:]=hr; X[1:,0]=hr
        # 3) t halfspaces: sum_i X[i+1,i+s+1] + t >= N/2
        # adjust t to satisfy all: t >= max_s (N/2 - sum_i X[i+1,i+s+1])
        need=0
        for s in range(N):
            val=np.sum(X[1+s:1+s+(N-s), 1:1+(N-s)])  # X[i+1, i+s+1] for i=0..N-1-s
            # i+1 in 1..N-s, i+s+1 in s+1..N
            need=max(need, N/2 - val)
        t=max(t, need)
        # 4) project onto PSD: eigenvalue clip
        w,V=np.linalg.eigh(X)
        w=np.maximum(w,0)
        X=V@np.diag(w)@V.T
        # re-symmetrize
        X=(X+X.T)/2
    # extract h
    h=X[0,1:].copy()
    h=project(h)
    return h, t
