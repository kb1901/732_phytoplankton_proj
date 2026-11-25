"""Forcing functions: each factory returns f(t) with t in SECONDS (times in the arguments are days).
Several can be combined for one run with add(...)."""
import numpy as np

DAY = 86400.0

def constant(v):
    return lambda t: v

def sine(mean, amp, period_days=365.0, phase_days=0.0):
    """mean + amp*sin(2*pi*(t - phase)/period)."""
    return lambda t: mean + amp * np.sin(2 * np.pi * (t / DAY - phase_days) / period_days)

def pulses(times_days, amp, width_days=2.0):
    """Gaussian pulses (storms, discharge events) of height amp at the given days."""
    tc = np.asarray(times_days, float)
    return lambda t: amp * np.sum(np.exp(-0.5 * ((t / DAY - tc) / width_days) ** 2))

def seasonal_flux(mean, amp, peak_days=240.0, width_days=45.0, period_days=365.0):
    """ mean + amp * np.exp(-tan^2(mean*x/period) / tan^2(m)) """
    return lambda t: mean + amp * np.exp(-(np.tan(np.pi * ((t/DAY) - peak_days) / period_days)**2) / np.tan(np.pi * width_days / period_days)**2)

def ramp(t0_days, t1_days, v0, v1):
    """Linear trend from v0 to v1 between two days (flat outside)."""
    return lambda t: v0 + (v1 - v0) * np.clip((t / DAY - t0_days) / (t1_days - t0_days), 0, 1)

def add(*fs):
    """Sum of forcings: add(sine(...), pulses(...), ramp(...))."""
    return lambda t: sum(f(t) for f in fs)
