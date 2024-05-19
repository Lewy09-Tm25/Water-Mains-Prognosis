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


tasks = [
    Task(
        priority=8,  # = pipe condition score
        n_work_hours=2,  # time depends on length of segment
        n_segment=None,
    ),
    Task(
        priority=4,  # = pipe condition score
        n_work_hours=11,
        n_segment=None,
    ),
    Task(
        priority=2,  # = pipe condition score
        n_work_hours=3,
        n_segment=None,
    ),
    Task(
        priority=1,  # = pipe condition score
        n_work_hours=5,
        n_segment=None,
    ),
]

hourly_tasks = [HourlyTask(task) for task in tasks for i in range(task.n_work_hours)]
