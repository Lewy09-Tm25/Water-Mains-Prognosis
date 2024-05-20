import numpy as np
from csp import (
    CSP,
    TravelConstraint,
    SupervisorConstraint,
    WeatherConstraint,
)
from plot import plot
import time

from scheduler.core import Timetable
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
    result_schedule = csp.solve(hourly_tasks=hourly_tasks, tasks=tasks).array
    return result_schedule


if __name__ == "__main__":
    # Execution time
    # n_runs = 100
    # total_time = 0.0
    # for _ in range(n_runs):
    #     start = time.perf_counter()
    #     _ = _run_scheduling_process()  # Call the function to run the process
    #     end = time.perf_counter()
    #     total_time += (end - start)
    #
    # average_time = total_time / n_runs
    # print(f"Average execution time over {n_runs} runs: {average_time:.6f} seconds")
    #
    result_schedule = _run_scheduling_process()
    plot(schedule=result_schedule, employees=employees)
