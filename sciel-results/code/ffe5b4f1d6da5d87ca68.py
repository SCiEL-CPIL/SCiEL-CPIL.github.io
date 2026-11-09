x = np.arange(1, 10*K+1)
# This is hard to vectorize easily without broadcasting to 10K x m
# Or loop over x:
for x in range(1, 10*K+1):
    # sum f[k] * (x // k)
