import math


def calculate_physics(quantity: str, **values) -> dict:
    """Calculate a physics quantity with explicit SI inputs."""
    formulas = {
        'force': (('mass', 'acceleration'), lambda x: x['mass'] * x['acceleration'], 'N', 'F = m * a'),
        'kinetic_energy': (('mass', 'velocity'), lambda x: 0.5 * x['mass'] * x['velocity'] ** 2, 'J', 'Ek = m * v² / 2'),
        'momentum': (('mass', 'velocity'), lambda x: x['mass'] * x['velocity'], 'kg·m/s', 'p = m * v'),
        'speed': (('distance', 'time'), lambda x: x['distance'] / x['time'], 'm/s', 'v = s / t'),
        'work': (('force', 'distance'), lambda x: x['force'] * x['distance'], 'J', 'A = F * s (force parallel to displacement)'),
        'density': (('mass', 'volume'), lambda x: x['mass'] / x['volume'], 'kg/m³', 'ρ = m / V'),
    }
    if quantity not in formulas:
        return {'error': f'Unknown quantity: {quantity}'}
    required, fn, unit, formula = formulas[quantity]
    try:
        x = {name: float(values[name]) for name in required}
        if not all(math.isfinite(v) for v in x.values()):
            raise ValueError('Inputs must be finite numbers')
        if 'mass' in x and x['mass'] < 0:
            raise ValueError('Mass cannot be negative')
        if 'distance' in x and x['distance'] < 0:
            raise ValueError('Distance cannot be negative')
        if 'time' in x and x['time'] <= 0:
            raise ValueError('Time must be positive')
        if 'volume' in x and x['volume'] <= 0:
            raise ValueError('Volume must be positive')
        result = fn(x)
        if not math.isfinite(result):
            raise ValueError('Result is not finite')
        return {'quantity': quantity, 'inputs': x, 'formula': formula, 'result': round(result, 8), 'unit': unit}
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        return {'error': str(exc)}
