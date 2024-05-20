from time import perf_counter

import numpy as np
from csp import CSP, SupervisorConstraint, TravelConstraint, WeatherConstraint
from plot import plot
from datetime import date, datetime, time, timedelta

from scheduler.core import Timetable, save_gantt_chart_tasks
from scheduler.workforce import employees, hourly_tasks, tasks

bad_weather_on_monday = np.zeros((7, 25))
bad_weather_on_monday[0] = np.ones(25)

constraints = [
    WeatherConstraint(table=Timetable(array=bad_weather_on_monday)),
    SupervisorConstraint(employees=employees),
    TravelConstraint(employees=employees),
]


def _run_scheduling_process():
    tt = Timetable.from_employee_list(employees=employees)
    csp = CSP(time_table=tt, constraints=constraints)
    return csp.solve(hourly_tasks=hourly_tasks, tasks=tasks).array


def get_next_monday() -> datetime:
    today = date.today()
    delta = 7 - today.weekday()
    return datetime.combine(today + timedelta(days=delta), time(hour=0, minute=0))


if __name__ == "__main__":
    # Execution time
    # n_runs = 100
    # total_time = 0.0
    # for _ in range(n_runs):
    #     start = perf_counter()
    #     _ = _run_scheduling_process()  # Call the function to run the process
    #     end = perf_counter()
    #     total_time += (end - start)
    #
    # average_time = total_time / n_runs
    # print(f"Average execution time over {n_runs} runs: {average_time:.6f} seconds")
    #
    result_schedule = _run_scheduling_process()

    schedule = []
    next_monday = get_next_monday()

    for emp_idx, employee in enumerate(employees):
        emp_schedule = result_schedule[:, :, emp_idx]  # (7, 25, n_tasks)

        item = [
            dict(
                Task=employee.name,
                Start=next_monday
                + timedelta(days=int(non_zero_idx[0]), hours=int(non_zero_idx[1])),
                Finish=next_monday
                + timedelta(days=int(non_zero_idx[0]), hours=int(non_zero_idx[1]) + 1),
                Resource="Segment {}".format(tasks[non_zero_idx[-1]].n_segment),
            )
            for non_zero_idx in zip(*np.nonzero(emp_schedule))
        ]
        schedule += item

    plot(schedule=schedule)

    save_gantt_chart_tasks(schedule)
