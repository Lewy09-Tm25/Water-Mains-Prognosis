from datetime import date, datetime, time, timedelta

import numpy as np
import plotly.figure_factory as ff

from csp import Employee


def get_next_monday() -> datetime:
    today = date.today()
    delta = 7 - today.weekday()
    return datetime.combine(today + timedelta(days=delta), time(hour=0, minute=0))


def plot(schedule: np.ndarray, employees: list[Employee]) -> None:
    schedule_df = []
    next_monday = get_next_monday()

    for emp_idx, employee in enumerate(employees):
        emp_schedule = schedule[:, :, emp_idx]  # (7, 25, n_tasks)

        df = [
            dict(
                Task=employee.name,
                Start=next_monday
                + timedelta(days=int(non_zero_idx[0]), hours=int(non_zero_idx[1])),
                Finish=next_monday
                + timedelta(days=int(non_zero_idx[0]), hours=int(non_zero_idx[1]) + 1),
                Resource=f"Task {non_zero_idx[-1]}",
            )
            for non_zero_idx in zip(*np.nonzero(emp_schedule))
        ]
        schedule_df += df

    fig = ff.create_gantt(
        schedule_df,
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
