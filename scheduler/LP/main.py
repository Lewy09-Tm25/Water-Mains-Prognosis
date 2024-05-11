from pulp import LpProblem, LpVariable, lpSum

from scheduler.workforce import employees, hourly_tasks


if __name__ == "__main__":
    prob = LpProblem()

    schedule = LpVariable.dicts(
        "Schedule",
        [
            (emp.name, hourly_task.task.id)
            for emp in employees
            for hourly_task in hourly_tasks
        ],
        cat="Binary",
    )

    prob += lpSum(
        schedule[(emp.name, hourly_task.task.id)]
        for emp in employees
        for hourly_task in hourly_tasks
    )

    # Constraints: Workers cannot exceed their working hours, assuming that their schedule is flexible
    for emp in employees:
        prob += lpSum(schedule[(emp.name, hourly_task.task.id)] for hourly_task in hourly_tasks) <= (len(emp.hours))

    # Constraint: Workers require supervisor on site

    prob.solve()

    print(prob.objective)
