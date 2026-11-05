    def try_combo(combo):
        nonlocal best_deficit, best_indices
        s = Fraction(0)
        for i in combo:
            s += recips_frac[i]
        if s >= 1:
            return
        deficit = 1 - s
        if best_deficit is None or deficit < best_deficit:
            best_deficit = deficit
            best_indices = list(combo)
