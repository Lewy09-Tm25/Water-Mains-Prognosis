from pulp import LpProblem, LpVariable, lpSum

from scheduler.workforce import employees, tasks


if __name__ == "__main__":
    prob = LpProblem()

    # Define binary variables for each task and worker
    schedule = LpVariable.dicts(
        "Schedule",
        [
            (task.id, emp.name)
            for task in tasks
            for emp in employees
        ],
        cat="Binary",
    )

    # Define the objective function
    prob += lpSum(
        task.n_work_hours * schedule[(task.id, emp.name)]
        for task in tasks
        for emp in employees
    )

    # Constraints: Each task must be assigned to exactly one worker
    for task in tasks:
        prob += lpSum(schedule[(task.id, emp.name)] for emp in employees) == 1

    # Constraints: Workers cannot exceed their working hours, assuming that their schedule is flexible
    for emp in employees:
        prob += lpSum(
            task.n_work_hours * schedule[(task.id, emp.name)]
            for task in tasks
        ) <= len(emp.hours)

    # Constraint: Workers require supervisor on site
    for task in tasks:
        supervisor_assigned = False
        for emp in employees:
            if emp.role == 'Supervisor':
                supervisor_assigned = supervisor_assigned or schedule[(task.id, emp.name)]
        prob += supervisor_assigned == 1

    prob.solve()

    print(prob.objective)
