    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        return Om0 * a**3 / H2(a, Om0, w0, wa)
