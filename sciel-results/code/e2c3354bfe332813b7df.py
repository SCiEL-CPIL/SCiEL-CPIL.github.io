    t = t_start
    while t < t_end:
        h_step = min(h, t_end - t)
        k1 = lorenz(t, y)
        k2 = lorenz(t + h_step/2, y + h_step/2 * k1)
        k3 = lorenz(t + h_step/2, y + h_step/2 * k2)
        k4 = lorenz(t + h_step, y + h_step * k3)
        y = y + h_step/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += h_step
