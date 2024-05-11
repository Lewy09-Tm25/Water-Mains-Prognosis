from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count
from typing import Generator, Literal, NoReturn, Union

import numpy as np

DAYS_INDICES = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


@dataclass
class Employee:
    name: str
    hours: dict[str, list[int]]
    role: Literal["Supervisor", "Worker"] = "Worker"


@dataclass
class Timetable:
    array: np.ndarray = np.zeros((7, 25))

    @classmethod
    def from_employee_list(cls, employees: list[Employee]) -> "Timetable":
        new_array = np.zeros((7, 25, len(employees)))
        for emp_idx, employee in enumerate(employees):
            for day in employee.hours.keys():
                new_array[DAYS_INDICES[day], employee.hours[day], emp_idx] = 1
        return cls(array=new_array)

    def slots(self) -> Generator[tuple[int, int], None, None]:
        for non_zero_indices in zip(*np.nonzero(self.array)):
            yield non_zero_indices


@dataclass
class Task:
    priority: int
    n_work_hours: int
    n_segment: int
    id: int = field(default_factory=count().__next__)


@dataclass
class HourlyTask:
    task: Task


@dataclass
class Segment:
    n_segment: int
    material: str
    diameter: int
    length: int
    preasure: float


@dataclass
class Constraint(ABC):
    """
    An abstract class that represents a constraint for CSP.
    """

    @abstractmethod
    def eval(self, slot: tuple[int, int], **kwargs) -> bool:
        """
        An abstract method to test if the constraint holds.
        """
        pass


@dataclass
class WeatherConstraint(Constraint):
    """
    Workers may only work during days without rain.
    """

    table: Timetable  # rain = 1

    def eval(self, slot: tuple[int, int], **kwargs) -> bool:
        day, hour = slot[:2]
        return self.table.array[day, hour] == 1


@dataclass
class SupervisorConstraint(Constraint):
    """
    Workers may only work when a supervisor is on site.
    """

    employees: list[Employee]

    def __post_init__(self) -> None:
        self._supervisor_indices = [
            idx for idx, emp in enumerate(self.employees) if emp.role == "Supervisor"
        ]

    def eval(self, **kwargs) -> bool:
        slot = kwargs["slot"]
        if slot[-1] in self._supervisor_indices:  # always try to schedule a supervisor
            return False

        day, hour = slot[:2]
        task_id = kwargs["task"].task.id
        return (
            1
            not in kwargs["schedule"].array[
                day, hour, self._supervisor_indices, task_id
            ]
        )


@dataclass
class EmployeeTravelTimeConstraint(Constraint):
    """
    Workers need at least 1 hour to travel to a new repairment site.
    """

    def eval(self, **kwargs) -> bool:
        return False
        # day, hour, emp_idx = kwargs['slot'][:3]
        # task_id = kwargs['task'].task.id

        # if hour - 1 < 0:
        #   day = day - 1
        # else:
        #   hour = hour - 1

        # try:
        #   scheduled_for_other_tasks = kwargs['schedule'].array[day, hour, emp_idx, ~task_id]
        #   if type(scheduled_for_other_tasks) == np.float64:
        #     return bool(scheduled_for_other_tasks)
        #   return 1 not in kwargs['schedule'].array[day, hour, emp_idx, ~task_id]
        # except IndexError:
        #   return False


@dataclass
class CSP:
    time_table: Timetable
    constraints: list[Constraint]

    def is_consistent(self, task, schedule):
        return len(np.nonzero(np.sum(schedule.array, axis=-1))) > 0

    def backtracking(
        self,
        tasks_to_schedule: list[HourlyTask],
        unassigned: list[HourlyTask],
        schedule: Timetable,
    ):
        if len(unassigned) == 0:
            return schedule

        task = unassigned.pop(0)

        for slot in self.time_table.slots():
            # Check for consistency
            if not self.is_consistent(task, schedule):
                continue

            # Check for constraints
            if any(
                constraint.eval(
                    slot=slot, time_table=self.time_table, schedule=schedule, task=task
                )
                for constraint in self.constraints
            ):
                continue

            # Assign available worker slot to schedule
            self.time_table.array[slot] = 0
            schedule.array[slot + (task.task.id,)] = 1

            result = self.backtracking(
                tasks_to_schedule,
                unassigned.copy(),
                Timetable(array=schedule.array.copy()),
            )
            if result:
                return result

            # Unassign worker slot
            self.time_table.array[slot] = 1
            schedule.array[slot + (task.task.id,)] = 0

    def solve(
        self, tasks: list[Task], hourly_tasks: list[HourlyTask]
    ) -> Union[Timetable, NoReturn]:
        empty_schedule = Timetable(
            array=np.zeros(self.time_table.array.shape + (len(tasks),))
        )
        result = self.backtracking(
            tasks_to_schedule=hourly_tasks,
            unassigned=hourly_tasks,
            schedule=empty_schedule,
        )
        if not result:
            raise Exception("No solution found for set of tasks.")
        return result
