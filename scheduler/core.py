from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count
from typing import Generator, Literal

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
