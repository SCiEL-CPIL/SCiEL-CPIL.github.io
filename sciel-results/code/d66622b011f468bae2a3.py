        k = np.empty((n_stages, len(y)))
        for j in range(n_stages):
            if j == 0:
                y_stage = y
            else:
                y_stage = y + h * np.dot(A[j, :j], k[:j])
            k[j] = f(t + c[j] * h, y_stage)
        
        y = y + h * np.dot(b, k)
