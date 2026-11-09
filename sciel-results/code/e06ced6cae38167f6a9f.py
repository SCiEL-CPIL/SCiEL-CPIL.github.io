    def _score_classes_fast(classes, base=51975):
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(covered == 0))
    