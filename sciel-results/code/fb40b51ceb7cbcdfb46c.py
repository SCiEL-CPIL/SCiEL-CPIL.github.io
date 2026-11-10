    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        arr_sq = (arr ** 2).sum(axis=1)
        p_sq = (p ** 2).sum()
        v = arr - p
        s = arr_sq - p_sq
        
        for combo in itertools.combinations(range(m), 4):
            idx = combo
            mat = np.hstack((v[idx], s[idx][:, None]))
            if abs(np.linalg.det(mat)) < 0.5:
                return False
        return True
    