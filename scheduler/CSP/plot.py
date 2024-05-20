from datetime import date, datetime, time, timedelta

import plotly.figure_factory as ff


def get_next_monday() -> datetime:
    today = date.today()
    delta = 7 - today.weekday()
    return datetime.combine(today + timedelta(days=delta), time(hour=0, minute=0))


def plot(schedule: list) -> None:
    next_monday = get_next_monday()

    fig = ff.create_gantt(
        schedule,
        title="Schedule",
        group_tasks=True,
        show_colorbar=True,
        index_col="Resource",
    )
    fig.update_layout(
        xaxis={"rangeselector": {"visible": False}},
        xaxis_range=[
            next_monday.strftime("%Y-%m-%d"),
            (next_monday + timedelta(days=6)).strftime("%Y-%m-%d"),
        ],
    )
    fig.show()
