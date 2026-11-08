"""fake arm so the others can work without the hardware. same methods the
real one will have, just prints and sleeps. swap in RealArm later."""

import time


class StubArm:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.log = []

    def pick_bottle(self, pill):
        self._do(f"pick bottle {pill}")

    def dispense_to_tray(self, count):
        self._do(f"tip {count} onto tray")

    def place_in_cell(self, day, slot):
        self._do(f"place pill in {day}/{slot}")

    def home(self):
        self._do("home")

    def _do(self, msg):
        # dashboard reads this list
        print(f"[arm] {msg}")
        self.log.append(msg)
        time.sleep(self.delay)


class RealArm(StubArm):
    """todo once I have a checkpoint. each method loads it and runs the
    policy with the task string for that cell."""

    def place_in_cell(self, day, slot):
        raise NotImplementedError("wire up lerobot policy here")