def objective(hh, tau=0.005):
    h = full_from_half(hh)
    conv = np.convolve(h, h, mode='full')
    cmax = conv.max()
    # guard against overflow
    soft = cmax + tau * np.log(np.sum(np.exp((conv - cmax)/tau)))
    max_h = h.max()
    sum_h = h.sum()
    if max_h < 1e-12 or sum_h < 1e-12:
        return 1e9
    return soft / (max_h * sum_h)
