def equation(x, y, params):
    return (params[0] * 1 +
            params[1] * x +
            params[2] * y +
            params[3] * x**2 +
            params[4] * x * y +
            params[5] * y**2 +
            params[6] * x**3 +
            params[7] * x**2 * y +
            params[8] * x * y**2 +
            params[9] * y**3)
