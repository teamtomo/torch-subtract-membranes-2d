from datetime import timedelta


def humanize_timedelta(td: timedelta) -> str:
    """Compact human-readable timedelta (e.g., '2d 5h')."""
    if isinstance(td, timedelta):
        total_seconds = int(td.total_seconds())
    else:
        total_seconds = int(td)

    if total_seconds == 0:
        return "0s"

    negative = total_seconds < 0
    total_seconds = abs(total_seconds)

    units = [
        ('y', 365 * 24 * 3600),
        ('mo', 30 * 24 * 3600),
        ('w', 7 * 24 * 3600),
        ('d', 24 * 3600),
        ('h', 3600),
        ('m', 60),
        ('s', 1)
    ]

    parts = []
    for unit_name, unit_seconds in units:
        if total_seconds >= unit_seconds:
            count = total_seconds // unit_seconds
            total_seconds %= unit_seconds
            parts.append(f"{count}{unit_name}")

            if len(parts) >= 2:  # Show max 2 units
                break

    result = "".join(parts)
    return f"-{result}" if negative else result
