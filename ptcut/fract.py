#! /usr/bin/env python3

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
try:
    from sage.all_cmdline import *                      # import sage library
    IS_SAGE = True
except ImportError:
    IS_SAGE = False


if IS_SAGE:
    class fract(Rational):
        def __init__(self, n, d):
            super(fract, self).__init__((n,d))

else:
    from fractions import Fraction
    class fract(Fraction):
        def numer(self):
            return self.numerator
        def denom(self):
            return self.denominator
