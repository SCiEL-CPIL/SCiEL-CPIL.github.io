        cum = np.zeros(n)
        if n > 1:
            cum[-2] = trap[-1]
            for i in range(n-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
