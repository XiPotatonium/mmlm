from typing import Any, Optional, Tuple
from rich.progress import (
    Progress, GetTimeCallable, TaskID, ProgressColumn, MofNCompleteColumn,
    Task, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
)


def get_default_columns() -> Tuple[ProgressColumn, ...]:
    """Get the default columns used for a new Progress instance:
        - a text column for the description (TextColumn)
        - the bar itself (BarColumn)
        - a text column showing completion percentage (TextColumn)
        - an estimated-time-remaining column (TimeRemainingColumn)
    If the Progress instance is created without passing a columns argument,
    the default columns defined here will be used.

    You can also create a Progress instance using custom columns before
    and/or after the defaults, as in this example:

        progress = Progress(
            SpinnerColumn(),
            *Progress.default_columns(),
            "Elapsed:",
            TimeElapsedColumn(),
        )

    This code shows the creation of a Progress display, containing
    a spinner to the left, the default columns, and a labeled elapsed
    time column.
    """
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )

def no_total_columns() -> Tuple[ProgressColumn, ...]:
    return (
        SpinnerColumn(),
        TextColumn("{task.description}: {task.completed}"),
        TimeElapsedColumn(),
    )

def full_columns() -> Tuple[ProgressColumn, ...]:
    """这是我自己觉得比较舒服的组合
    """
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
