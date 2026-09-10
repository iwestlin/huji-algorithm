from enum import Enum


class JobType(Enum):
    SERVICE = "service"

    @staticmethod
    def from_str(value: str) -> "JobType":
        return JobType[value.upper()]
