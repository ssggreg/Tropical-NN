
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

from __future__ import print_function
import sys


class Prt:
    def __init__(self):
        self.log_file = None
    def __call__(self, *args, **kwargs):
        # python 2 doesn't allow the *args to be in front of named args with default values,
        # but putting it at the end will cause prt(1,2) to set "end" to 2.
        end = kwargs.pop("end", "\n")
        flush = kwargs.pop("flush", False)
        screen = kwargs.pop("screen", True)
        log = kwargs.pop("log", True)
        flushfile = kwargs.pop("flushfile", False)
        s = ""
        for arg in args:
            if s:
                s += " "
            s += str(arg)
        s += end
        if screen:
            print(end=s)
        if self.log_file:
            if log:
                print(end=s, file=self.log_file)
            if flushfile:
                self.log_file.flush()
        if flush:
            sys.stdout.flush()
    def set_log_file(self, file):
        self.log_file = file
    def get_log_file(self):
        return self.log_file
    def close_log_file(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    def flush_log_file(self):
        if self.log_file:
            self.log_file.flush()
    def flush_all(self):
        sys.stdout.flush()
        self.flush_log_file()


# global instance
prt = Prt()


if __name__ == "__main__":
    prt(1,2,3)
