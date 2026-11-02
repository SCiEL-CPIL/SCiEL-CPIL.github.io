def autoconv_fft(x):
    n = len(x)
    size = 2 * n - 1
    f1 = np.fft.rfft(x, size)
    f2 = np.fft.rfft(x, size)
    c = np.fft.irfft(f1 * f2, size)
    return c
