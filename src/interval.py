from typing import Final, Optional

import numpy as np
from deal import pre, inv


#@inv(lambda self: (self.lower is not None and self.upper is not None) != self.empty)
class Interval:
    lower: Optional[np.float64] = None
    upper: Optional[np.float64] = None
    empty: bool = True

    @pre(lambda _, empty, lower, upper: empty or lower <= upper)
    @pre(lambda _, empty, lower, upper: empty != (lower is not None and upper is not None))
    def __init__(self, empty: bool, lower: Optional[np.float64], upper: Optional[np.float64]):
        self.empty = empty
        self.lower: Optional[np.float64] = lower
        self.upper: Optional[np.float64] = upper

    def size(self) -> Optional[np.float64]:
        if self.empty:
            return None

        return self.upper - self.lower

    def clamp(self, x: np.float64) -> Optional[np.float64]:
        if self.empty:
            return None

        return np.clip(x, self.lower, self.upper)

    @pre(lambda _, x: np.isfinite(x))
    def __contains__(self, x: np.float64) -> bool:
        return self.lower <= x <= self.upper

    @pre(lambda _, x: np.isfinite(x))
    def surrounds(self, x: np.float64) -> bool:
        return self.lower < x < self.upper


empty: Final[Interval] = Interval(True, None, None)
universe: Final[Interval] = Interval(False, np.float64(-np.inf), np.float64(np.inf))
