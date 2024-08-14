from .TaskManager import TaskManager
from .ICLManager import ICLManager


def create_task(config):
    """
    Factory function to create the data manager object.
    @Note: TaskManager[i] returns the raw HTML list and label, while
    ICLManager[i] returns the parsed data and label.
    """
    return TaskManager(config), ICLManager(config)