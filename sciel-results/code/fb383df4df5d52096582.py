    try:
        B = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        # If C is not positive definite, reset or break
        break
    