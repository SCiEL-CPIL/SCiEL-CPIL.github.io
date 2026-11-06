import inspect
import sys

# Check if the module is available and inspect the baseline
try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    print("Baseline source:")
    print(inspect.getsource(pebble_baseline))
except Exception as e:
    print(f"Error: {e}")
    # Try to find the module
    try:
        import autoresearch_suite.science_tcs_expansion as mod
        print("Module file:", mod.__file__)
        print(dir(mod))
    except Exception as e2:
        print(f"Module error: {e2}")
