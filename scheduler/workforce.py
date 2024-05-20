import random
from pathlib import Path

import pandas as pd

from scheduler.core import Employee, HourlyTask, Task

employees = [
    Employee(
        name="Alice",
        hours={
            "Tuesday": [10, 11, 12, 14, 15, 16],
            "Wednesday": [10, 11, 12, 14, 15, 16],
            "Friday": [10, 11, 12, 14, 15, 16],
            "Saturday": [10, 11, 12, 14],
        },
        role="Supervisor",
    ),
    Employee(
        name="Emily",
        hours={
            "Monday": [10, 11, 12, 14, 15, 16],
            "Tuesday": [10, 11, 12, 14, 15, 16],
            "Wednesday": [10, 11, 12, 14, 15, 16],
        },
    ),
    Employee(
        name="Bob",
        hours={
            "Tuesday": [10, 11, 12, 14, 15, 16],
            "Wednesday": [10, 11, 12, 14, 15, 16],
        },
    ),
]


df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'csp_data.csv')
df = df.sort_values(by='Predicted condition', ascending=True)

tasks = [
    Task(
        priority=10-row['Predicted condition'],  # The lower the score, the more urgent the repair
        n_work_hours=random.randint(2, 10),  # TODO: time should depend on pipe segment length
        n_segment=row['id_index']
    )
    for _, row in df[:4].iterrows()
]

hourly_tasks = [HourlyTask(task) for task in tasks for i in range(task.n_work_hours)]
