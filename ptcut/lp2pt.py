
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

from phwrapper import *
import os


def dim_expand(l, maxlen):
    """
    >>> dim_expand([[1],[2]], 2)
    [(1, 0), (2, 0)]
    >>> dim_expand([[]], 2)
    [(0, 0)]
    >>> dim_expand([[1],[2,2]], 2)
    [(1, 0), (2, 2)]
    """
    r = []
    for i in l:
        r.append(tuple(i + [0] * (maxlen - len(i))))
    return r


def read_lp_file(f, rich=False):
    r"""
    Convert a .lp file to a list of input eqns and ieqs for phwrap.

    >>> read_lp_file([ "Subject To", "  ie0: -1 x1 >= -6", "  eq0: +1 x1 = -22", "BOUNDS" ])
    ([(22, 1)], [(6, -1)])
    >>> read_lp_file([ "Subject To", "  ie0: -1 x2 >= -6", "  ie0: +1 x1 +1 x3 >= 4", "  eq0: +1 x1 = -22", "  eq0: +1 x1 -1 x2 -2 x3 = +14", "BOUNDS" ])
    ([(22, 1, 0, 0), (-14, 1, -1, -2)], [(6, 0, -1, 0), (-4, 1, 0, 1)])
    """
    valid = False
    ign = True
    ie = []
    sie = []
    eq = []
    seq = []
    maxlen = 0
    for l in f:
        l = l.split("\\", 1)[0].strip()
        if l == "Subject To":
            ign = False
            valid = True
            continue
        if l == "BOUNDS":
            ign = True
            continue
        if l == "END":
            break
        if not l or ign:
            continue
        src = l
        a = l.split(":")
        if len(a) != 2:
            print("Invalid constraint line: {}".format(l))
            continue
        a = a[1].split("=")
        if len(a) != 2:
            print("Invalid sense for constraint: {}".format(l))
            continue
        iseq = True
        if a[0][-1] == ">":
            iseq = False
            a[0] = a[0][:len(a[0])-1]                   # snip off last char
        vec = [-int(a[1])]      # r.h.s.
        a = a[0].split()
        for c,v in zip(a[::2], a[1::2]):
            if v[0] != "x":
                print("Invalid variable name: {}".format(v))
                break
            nb = int(v[1:])
            if len(vec) <= nb:
                vec = vec + [0] * (nb - len(vec) + 1)   # fill with zeros
            vec[nb] = int(c)
        if iseq:
            eq.append(vec)
            if rich:
                seq.append(src)
        else:
            ie.append(vec)
            if rich:
                sie.append(src)
        maxlen = max(maxlen, len(vec))

    if not valid:
        return None
    eq = dim_expand(eq, maxlen)
    ie = dim_expand(ie, maxlen)
    if rich:
        return eq, seq, ie, sie
    else:
        return eq, ie


def convert(f, fname=""):
    r"""
    Convert a .lp file to a list of input eqns and ieqs for phwrap.

    >>> convert([ "Subject To", "  ie0: -1 x1 >= -6", "  eq0: +1 x1 = -22", "BOUNDS" ])
    '    # from ""\n    ie = [\n        (6, -1),  # ie0: -1 x1 >= -6\n    ]\n    eq = [\n        (22, 1),  # eq0: +1 x1 = -22\n    ]\n    l.append(phwrap(eqns=eq, ieqs=ie))\n'
    >>> convert([ "Subject To", "  ie0: -1 x2 >= -6", "  ie0: +1 x1 +1 x3 >= 4", "  eq0: +1 x1 = -22", "  eq0: +1 x1 -1 x2 -2 x3 = +14", "BOUNDS" ])
    '    # from ""\n    ie = [\n        (6, 0, -1, 0),  # ie0: -1 x2 >= -6\n        (-4, 1, 0, 1),  # ie0: +1 x1 +1 x3 >= 4\n    ]\n    eq = [\n        (22, 1, 0, 0),  # eq0: +1 x1 = -22\n        (-14, 1, -1, -2),  # eq0: +1 x1 -1 x2 -2 x3 = +14\n    ]\n    l.append(phwrap(eqns=eq, ieqs=ie))\n'
    """
    if fname is not None:
        r = '    # from "{}"\n'.format(fname)
    else:
        r = ""
    rr = read_lp_file(f, rich=True)
    if rr is None:
        return None
    eq, seq, ie, sie = rr
    r += "    ie = [\n"
    for i,s in zip(ie,sie):
        r += "        {},  # {}\n".format(i, s)
    r += "    ]\n"

    r += "    eq = [\n"
    for i,s in zip(eq,seq):
        r += "        {},  # {}\n".format(i, s)
    r += "    ]\n"
    r += "    l.append(phwrap(eqns=eq, ieqs=ie))\n"
    return r


def convert_multi_file(f, fname=""):
    r = '    # from "{}"\n'.format(fname)
    while True:
        r1 = convert(f, None)
        if r1 is None:
            # no more last multi file pieces
            break
        r += r1
    return r


def load_ph_from_lp(seq, l):
    """
    Read a sequence from an .lp file, create the polyhedron and append it to list 'l'.
    """
    r = read_lp_file(seq)
    if r is None:
        # not an .lp file
        return
    eqns, ieqs = r
    p = phwrap(eqns=eqns, ieqs=ieqs)
    if p is not None:
        l.append(p)


def load_ph_from_lp_wildcard(wild):
    """
    Cycle through the wildcard, open files and read in .lp files.
    Files that are obviously not .lp files are ignored.
    Returns a list of polyhedra.
    """
    import glob
    l = []
    for fname in glob.glob(wild):
        with open(fname) as f:
            load_ph_from_lp(f, l)
    return l


def load_ph_from_multi_lp(fname):
    """
    Open and read (multi-) .lp file.
    Returns a list of polyhedra.
    """
    from util import multi_lp_sep
    l = []
    with open(fname) as f:
        lines = []
        for ln in f:
            if ln.startswith(multi_lp_sep) and lines:
                # end of section, create polyhedron
                load_ph_from_lp(lines, l)
                lines = []
            else:
                # collect line
                lines.append(ln.rstrip())
        if lines:
            # seems like a non-multi .lp file, create polyhedron
            load_ph_from_lp(lines, l)
    return l


def load_ph_from_fname_list(files):
    """
    From a list of filenames or wildcards, load the polyhedra.
    """
    for fname in files:
        if os.path.isfile(fname):
            # combo .lp file or single .lp file
            b = load_ph_from_multi_lp(fname)
        else:
            # collection of .lp files
            wild = fname + "*.lp"
            b = load_ph_from_lp_wildcard(wild)
    return b


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    files = []
    multi = False

    import sys
    for i in sys.argv[1:]:
        if i[0] == "-":
            if i == "-m":
                multi = True
        else:
            files.append(i)

    for i in files:
        with open(i) as f:
            print(convert_multi_file(f, i) if multi else convert(f, i))
