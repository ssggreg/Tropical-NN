
# $+HEADER$
#
# Copyright 2017-2019 Christoph Lueders
#
# This file is part of the PtCut project: <http://wrogn.com/ptcut>
#
# PtCut is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PtCut is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PtCut.  If not, see <http://www.gnu.org/licenses/>.
#
# $-HEADER$

from __future__ import print_function, division


class vector:
    """
    tiny vector class as replacement for SAGE's vector.

    >>> v = vector([1,2,3])
    >>> v
    (1, 2, 3)
    >>> w = vector([1,1,1])
    >>> v + w
    (2, 3, 4)
    >>> v - w
    (0, 1, 2)
    >>> 2 * v
    (2, 4, 6)
    >>> v * 2
    (2, 4, 6)
    >>> v * 2.0
    (2.0, 4.0, 6.0)
    >>> v * w
    6
    >>> -v
    (-1, -2, -3)
    >>> +v
    (1, 2, 3)
    """
    def __init__(self, v):
        self.v = v
    def __add__(self, w):
        return vector([a+b for a,b in zip(self, w)])
    def __sub__(self, w):
        return vector([a-b for a,b in zip(self, w)])
    def __mul__(self, w):
        try:
            return sum(a*b for a,b in zip(self, w))
        except TypeError:
            return vector([a * w for a in self])
    def __rmul__(self, w):
        return vector([a * w for a in self])
    def __repr__(self):
        return "(" + ", ".join([str(i) for i in self]) + ")"
    def __neg__(self):
        return vector([-a for a in self])
    def __pos__(self):
        return vector([a for a in self])
    def __len__(self):
        """
        >>> len(vector([1,2,3]))
        3
        >>> len(vector([]))
        0
        """
        return len(self.v)
    def __getitem__(self, length):
        """
        >>> vector([1,2,3])[0]
        1
        >>> vector([1,2,3])[2]
        3
        >>> vector([1,2,3])[1:]
        (2, 3)
        >>> vector([1,2,3])[3]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range
        >>> vector([[1,2],[3,4]])[1,1,1]
        Traceback (most recent call last):
          ...
        TypeError: index must be int or slice
        """
        if type(length) == int:
            return self.v[length]
        if type(length) == slice:
            return vector(self.v[length])
        raise TypeError("index must be int or slice")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
