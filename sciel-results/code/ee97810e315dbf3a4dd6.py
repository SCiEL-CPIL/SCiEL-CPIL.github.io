    def learned_priority(vector, n):
        x = _extract_features(vector, n)
        score = np.dot(w, x) + b
        return float(score)
