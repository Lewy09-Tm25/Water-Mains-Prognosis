from datetime import date, datetime, time, timedelta
from time import perf_counter

import pandas as pd
import plotly.figure_factory as ff
from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum

from scheduler.workforce import employees, tasks

DAYS_INDICES = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6,
}


def get_next_monday() -> datetime:
    today = date.today()
    delta = 7 - today.weekday()
    return datetime.combine(today + timedelta(days=delta), time(hour=0, minute=0))


def plot(task_blocks_df: pd.DataFrame) -> None:
    next_monday = get_next_monday()

    plot_df_dict = [
        dict(
            Task='Entire Workforce',
            Start=next_monday + timedelta(days=int(row['Block'] // 24), hours=int(row['Block'] % 24)),
            Finish=next_monday + timedelta(days=int(row['Block'] // 24), hours=int(row['Block'] % 24 + 1)),
            Resource=f"Task {row['Task']}",
        )
        for _, row in task_blocks_df.iterrows()
    ]

    fig = ff.create_gantt(
        pd.DataFrame(plot_df_dict),
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


def _run_scheduling_process():
    hours_of_day = [
        datetime.combine(date.today() + timedelta(days=day_delta), time(hour=hour_delta))
        for day_delta in range(7)
        for hour_delta in range(24)
    ]

    schedule_df = pd.DataFrame({'slot': hours_of_day})
    schedule_df['availability'] = 0

    hours = [0] * 24 * 7
    for emp in employees:
        for day in emp.hours.keys():
            for hour in emp.hours.get(day, []):
                day_idx = DAYS_INDICES[day]
                if day_idx == 0:
                    hours[hour] += 1
                else:
                    hours[hour + day_idx * 24] += 1

    schedule_df['availability'] = hours

    tasks_df = pd.DataFrame(tasks)

    s = list(tasks_df['priority'])
    d = list(tasks_df['n_work_hours'])
    b = list(schedule_df['availability'])

    B = len(b)
    n = len(s)
    A = sum(b)

    prob = LpProblem("Schedule", LpMaximize)
    y = LpVariable.dicts('Block', [(i, t) for i in range(n) for t in range(B)], cat='Binary')
    prob += lpSum(s[i] * b[t] * y[(i, t)] for i in range(n) for t in range(B))

    # Constraints
    prob += lpSum(y[(i, t)] for i in range(n) for t in range(B)) <= A  # Total assigned hours

    for i in range(n):
        prob += lpSum(y[(i, t)] for t in range(B)) <= d[i]  # Task duration

    for t in range(B):
        prob += lpSum(y[(i, t)] for i in range(n)) <= schedule_df['availability'][t]  # Max tasks per slot

    prob.solve(PULP_CBC_CMD(msg=False))

    tasks_blocks = pd.DataFrame(columns=['Task', 'Block'])

    for i in range(n):
        for t in range(B):
            if y[(i, t)].varValue == 1:
                tasks_blocks = pd.concat([tasks_blocks, pd.DataFrame({'Task': [i], 'Block': [t]})], ignore_index=True)

    tasks_blocks['Task'] = tasks_blocks['Task'].astype(int)
    tasks_blocks['Block'] = tasks_blocks['Block'].astype(int)

    return tasks_blocks


if __name__ == "__main__":
    # Execution time
    # n_runs = 100
    # total_time = 0.0
    # for _ in range(n_runs):
    #     start = perf_counter()
    #     _ = _run_scheduling_process()  # Call the function to run the process
    #     end = perf_counter()
    #     total_time += (end - start)

    # average_time = total_time / n_runs
    # print(f"Average execution time over {n_runs} runs: {average_time:.6f} seconds")

    tasks_blocks = _run_scheduling_process()
    plot(tasks_blocks)
