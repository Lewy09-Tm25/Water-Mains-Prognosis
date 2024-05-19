import numpy as np
from csp import (
    CSP,
    TravelConstraint,
    SupervisorConstraint,
    WeatherConstraint,
)
from plot import plot

from scheduler.core import Timetable
from scheduler.workforce import employees, hourly_tasks, tasks

bad_weather_on_monday = np.zeros((7, 25))
bad_weather_on_monday[0] = np.ones(25)

constraints = [
    WeatherConstraint(table=Timetable(array=bad_weather_on_monday)),
    SupervisorConstraint(employees=employees),
    TravelConstraint(employees=employees),
]


if __name__ == "__main__":
    tt = Timetable.from_employee_list(employees=employees)

    csp = CSP(time_table=tt, constraints=constraints)
    result_schedule = csp.solve(hourly_tasks=hourly_tasks, tasks=tasks).array

    plot(schedule=result_schedule, employees=employees)
