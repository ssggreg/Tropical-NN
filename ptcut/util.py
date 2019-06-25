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

import numpy as np
from prt import prt
from phwrapper import phwrap
from math import log, log10, floor
import fractions
import sys
from fract import fract


#from os import times
#import resource

#def mytime():
#    return clock()
#    #t = os.times()
#    #return t[0] + t[1]
#    #r = resource.getrusage(0)
#    #return r[0] + r[1]

try:
    from time import perf_counter
    mytime = perf_counter
except ImportError:
    from time import clock
    mytime = clock

# using time.clock():
#    min=0.000 avg=0.863 max=38.000, dev=0.937 usec
# using os.times():
#    min=0.000 avg=1.100 max=10000.000, dev=104.876 usec
# using resource.getrusage():
#    min=0.000 avg=1.600 max=4000.000, dev=79.984 usec


def nlw_open(fname):
    """
    Open for write and use Unix line endings (LF), even under Windows.
    """
    try:
        # this should work under Python 3, but Python 2 doesn't know newline parameter
        return open(fname, "w", newline="\n")
    except TypeError:
        # Python 2 way that will not work for Python 3
        return open(fname, "wb")


inf = float("inf")

def prod(s):
    """
    >>> prod([1, 2, 3])
    6
    >>> prod([])
    1
    >>> prod([1, -1, 2, -2])
    4
    """
    r = 1
    for i in s:
        r *= i
    return r

avg = np.mean


def myround(v, n=0):
    """
    >>> myround(0.5)
    1
    >>> myround(1.5)
    2
    >>> myround(-0.5)
    -1
    >>> myround(-1.5)
    -2
    >>> myround(0.05, 1)
    0.1
    >>> myround(0.15, 1)
    0.2
    >>> myround(5, -1)
    10
    >>> myround(15, -1)
    20
    >>> myround(78124.99999999997, -5)
    100000
    >>> myround(24.999999999999996, -1)
    30
    """
    sign = -1 if v < 0 else 1
    v *= sign
    # handle FP-inaccuracy of 1 / 1e-05 == 99999.99999999999 (!)
    if n >= 0:
        scale = 10 ** n
        # handle FP-inaccuracy of 0.2 ** -2 == 24.999999999999996 (!)
        r = floor(v * scale + 0.5000000000000005) / scale
    else:
        scale = 10 ** -n
        r = floor(v / scale + 0.5000000000000005) * scale
    if type(v) == int or n <= 0:
        r = int(r)
    r *= sign
    return r


def toInt(a):
    return [[Integer(i) for i in v] for v in a]


def bestnum(s):
    """
    Convert string to number.  Use float for broken values and int for integers.

    >>> bestnum("1")
    1
    >>> bestnum("1.1")
    1.1
    >>> bestnum("1.0")
    1
    >>> bestnum("1.")
    1
    >>> bestnum("-1")
    -1
    >>> bestnum("0")
    0
    """
    while s != "0" and s[-1:] == "0":
        s = s[:-1]
    if s[-1:] == ".":
        s = s[:-1]
    try:
        return int(s)
    except ValueError:
        return float(s)


_prime_add = [4, 1, 1, 9, 7, 3, 3, 19, 7, 7, 19, 3, 39, 37, 31, 37, 61, 3, 3, 51, 39, 117, 9, 117, 7, 13, 67, 103, 331, 319,
    57, 33, 49, 61, 193, 69, 67, 43, 133, 3, 121, 109, 63, 57, 31, 9, 121, 33, 193, 9, 151, 121, 327,
    171, 31, 21, 3, 279, 159, 19, 7, 93, 447, 121, 57, 49, 49, 49, 99, 9, 33, 273, 39, 79, 207, 129,
    133, 21, 93, 49, 129, 13, 391, 27, 261, 103, 151, 373, 181, 31, 289, 79, 399, 153, 97, 151, 127,
    469, 49, 289, 267, 3, 117, 129, 267, 3, 79, 3, 19, 457, 7, 139, 207, 99, 271, 79, 93, 279, 709,
    69, 79, 87, 1119, 3, 753, 237, 679, 283, 387, 459, 1113, 63, 169, 21, 7, 1143, 91, 283, 273, 513,
    13, 129, 13, 81, 91, 73, 9, 987, 247, 183, 67, 901, 13, 87, 453, 189, 451, 583, 151, 187, 303, 403,
    133, 217, 1527, 559, 267, 27, 1323, 37, 63, 193, 441, 337, 691, 339, 151, 723, 261, 979, 313, 217,
    231, 999, 37, 927, 721, 163, 183, 181, 253, 57, 1153, 357, 951, 21, 39, 1107, 553, 153, 357]

def make_prime(dig):
    r = 10 ** dig + _prime_add[dig]
    assert log10(r) >= dig and log10(r) < dig+1, "dig={}, r={}, log10(r)={}".format(dig, r, log10(r))
    return r


class status_print:
    """
    Do status prints, i.e., print over old statuses to implement counting up without
    consuming screen space.
    """
    def __init__(self):
        self.e = 0
    def print(self, text):
        ps = "\b" * self.e + text
        if len(text) < self.e:
            ps += " " * (self.e - len(text)) + "\b" * (self.e - len(text))
        print(end=ps)
        sys.stdout.flush()
        self.e = len(text)


progress = True

def set_progress(p):
    global progress
    progress = p


class tame_it:
    """
    Do something only every x seconds, like printing a status.
    """
    def __init__(self, timeout):
        self.to = timeout
        self.t1 = mytime()
    def go(self):
        if progress:
            t2 = mytime()
            if t2 - self.t1 >= self.to:
                self.t1 = t2
                return True
        return False


def comboname(combo):
    """
    >>> comboname({0:1, 1:3})
    '1-3'
    >>> comboname({1:3})
    'x-3'
    >>> comboname({})
    ''
    """
    s = ""
    if combo:
        for i in range(max(combo.keys()) + 1):
            if s:
                s += "-"
            if i in combo:
                s += str(combo[i])
            else:
                s += "x"
    return s


def prtexp(e):
    if e != 1:
        return "**{}".format(e)
    return ""


def sol_to_lp(p, vars, ep=None, scale=None, bounds=False):
    r"""
    Print a polyhedron in .lp file format.

    >>> p1 = phwrap(eqns=[[1,2,3]])
    >>> sol_to_lp(p1, ["x1","x2"], bounds=True)
    '\\ dimension 1; is not compact\nMAXIMIZE\nSubject To\n  eq0: +2 x1 +3 x2 = -1\nBOUNDS\n  x1 free\n  x2 free\nEND\n'

    >>> p2 = phwrap(eqns=[[-1,2,-3,9]], ieqs=[[5,4,3,3],[-8,-1,5,-2]])
    >>> sol_to_lp(p2, ["x1","x2","x3"], bounds=True)
    '\\ dimension 2; is not compact\nMAXIMIZE\nSubject To\n  ie0: -5 x1 +39 x2 >= 74\n  ie1: +5 x1 +6 x2 >= -8\n  eq0: +2 x1 -3 x2 +9 x3 = 1\nBOUNDS\n  x1 free\n  x2 free\n  x3 free\nEND\n'
    """
    assert len(p.Hrep()[0].vector()) == len(vars) + 1
    icnt = [0]                                          # ints are immutable, hence use list with one element
    ecnt = [0]
    si = [""]
    se = [""]
    padlen = None
    if ep:
        assert scale
        delta = ep ** (-0.5 / scale) - ep ** (0.5 / scale)  # factor of inaccuracy due to rounding of log
    if not scale:
        scale = 1
    for h in p.Hrep():
        cnt = icnt if h.is_inequality() else ecnt
        sx = si if h.is_inequality() else se
        ln = "  {}{}:".format("ie" if h.is_inequality() else "eq", cnt[0])
        cnt[0] += 1
        for i,c in enumerate(h.vector()[1:]):
            if c:
                ln += " {}{} {}".format("" if c < 0 else "+", c, vars[i])
        ln += " {} {}".format(">=" if h.is_inequality() else "=", -h.vector()[0])
        if ep:
            # try to pad all lines to the same length
            if padlen is None:
                padlen = len(ln) + 8
            if padlen > len(ln):
                ln += " " * (padlen - len(ln))
            ln += r" \ "[:-1]                           # add trailing backslash
            # find gcd
            coeffs = [c for c in h.vector()[1:] if c != 0]
            gcd = coeffs[0]
            for c in coeffs[1:]:
                gcd = fractions.gcd(gcd, c)
            gcd = abs(gcd)
            # print monomial
            first = True
            for i,c in enumerate(h.vector()[1:]):
                if c:
                    ln += " {}{}{}".format("" if first else "* ", vars[i], prtexp(c // gcd))
                    first = False
            val = float(ep ** (-h.vector()[0] / gcd))
            rprec = int(-floor(log10(val * delta)))
            val = myround(val, rprec)
            ln += " {} ".format("=" if not h.is_inequality() else "<=")
            if val < 1e5 and val == int(val):
                ln += "{:.0f}".format(val)
            else:
                ln += "{:g}".format(val)
            #ln += "    ({})".format(gcd)
        sx[0] += ln + "\n"
    s = ""
    if "combo" in p.__dict__:
        s = "\\ combo: {}\n".format(comboname(p.combo))
    s += "\\ dimension {}; is {}compact\n".format(p.dim(), "" if p.is_compact() else "not ")
    if p.has_v:
        if p.is_compact() and p.dim() > 0:
            s += "\\ volume: {}\n".format(p.volume())
        s += "\\ center: {}\n".format(p.center())
        s += "\\ vertices:\n"
        for v in p.vertices_tuple_list():
            s += "\\    {}\n".format(v)
        if p.lines():
            s += "\\ lines:\n"
            for v in p.lines_tuple_list():
                s += "\\    {}\n".format(v)
        if p.rays():
            s += "\\ rays:\n"
            for v in p.rays_tuple_list():
                s += "\\    {}\n".format(v)
    s += "MAXIMIZE\nSubject To\n"
    s += si[0] + se[0]
    if bounds:
        s += "BOUNDS\n" + "".join(["  {} free\n".format(i) for i in vars])
    s += "END\n"
    return s


def sol_to_lpfile(r, fname1, vars, ep=None, scale=None):
    import os
    file_cnt = 0
    tame = tame_it(2)
    sts = status_print()
    for p in r:
        fname = fname1 + "-sol-{:05d}.lp".format(file_cnt)
        with nlw_open(fname) as f:
            #prt("[{}]".format(file_cnt))
            f.write(sol_to_lp(p, vars, ep=ep, scale=scale))
        file_cnt += 1
        if tame.go():
            sts.print("({})".format(file_cnt))
    # write dummy file if no solutions
    if file_cnt == 0:
        fname = fname1 + "-sol-{:05d}.lp".format(file_cnt)
        with nlw_open(fname) as f:
            print("\\ no solution", file=f)
        file_cnt += 1
    # remove old, maybe still existing files
    while True:
        fname = fname1 + "-sol-{:05d}.lp".format(file_cnt)
        if not os.path.isfile(fname):
            break
        #prt("<{}>".format(file_cnt))
        os.remove(fname)
        file_cnt += 1
        if tame.go():
            sts.print("({})".format(file_cnt))
    sts.print("")


multi_lp_sep = "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"

def sol_to_string(r, vars, ep=None, scale=None):
    s = ""
    file_cnt = 0
    tame = tame_it(2)
    sts = status_print()
    for p in r:
        s += "\n{} file {}\n\n".format(multi_lp_sep, file_cnt)
        s += sol_to_lp(p, vars, ep=ep, scale=scale)
        file_cnt += 1
        if tame.go():
            sts.print("({})".format(file_cnt))
    if not s:
        s = "\\ no solution"
    else:
        s += "\n{} end\n\n".format(multi_lp_sep)
    sts.print("")
    return s


def sol_string_to_one_lpfile(s, fname1):
    fname = fname1 + "-solutions.txt"
    with nlw_open(fname) as f:
        f.write(s)


def combo_cmp(a, b):
    """
    Compare the combo values a and b and return True if a has smaller values.

    >>> a = {0: 1, 1: 2}
    >>> b = {0: 0, 1: 2}
    >>> combo_cmp(a, b)
    1
    >>> combo_cmp(b, a)
    -1
    >>> combo_cmp(a, a)
    0
    >>> b = {0: 1, 1: 3}
    >>> combo_cmp(a, b)
    -1
    >>> combo_cmp(b, a)
    1
    >>> b = {0: 1, 2: 3}
    >>> combo_cmp(a, b)
    -1
    >>> combo_cmp(b, a)
    1
    >>> b = {2: 3}
    >>> combo_cmp(a, b)
    -1
    >>> combo_cmp(b, a)
    1
    >>> combo_cmp(b, b)
    0
    >>> a = {1: 1}
    >>> b = {1: 0}
    >>> combo_cmp(a, b)
    1
    >>> combo_cmp(b, a)
    -1
    """
    mx = max(max(a.keys()), max(b.keys()))
    for i in range(mx+1):
        if i in a and i in b:
            # both are listed.  return True if a is smaller, False if larger, keep going if equal.
            if a[i] != b[i]:
                return -1 if a[i] < b[i] else 1
        elif i in a or i in b:
            # only one is listed.  return True if a is listed, False if b is listed
            return -1 if i in a else 1
    return 0


def canon_repr(p):
    eqns = sorted(p.equalities_list())
    ieqs = sorted(p.inequalities_list())
    return eqns, ieqs


def canonicalize(rs):
    """
    Sort the equations and inequalities of each polyhedron and
    all the polyhedra against each other.
    """
    rs2 = []
    for p in rs:
        eqns, ieqs = canon_repr(p)
        p2 = phwrap(eqns=eqns, ieqs=ieqs)
        p2.combo = p.combo
        rs2.append(p2)
    return sorted(rs2, key=lambda x: canon_repr(x))


def compare_solutions(ps, qs):
    """
    Test if two sets of solutions are the same.
    """
    # first, compare space dimension
    if len(ps) > 0 and len(qs) > 0:
        p = next(iter(ps))
        q = next(iter(qs))
        if p.space_dim() != q.space_dim():
            return max(len(ps), len(qs))
    # create a work copy that we can consume
    ps = ps[:]
    qs = qs[:]
    # search each member of p in q.  if found, remove from both
    for p in ps[:]:
        for q in qs[:]:
            if p == q:
                ps.remove(p)
                qs.remove(q)
                break
    # search each member of q in p.  if found, remove from both
    for q in qs[:]:
        for p in ps[:]:
            if p == q:
                ps.remove(p)
                qs.remove(q)
                break
    # what's left now can not be matched
    return max(len(ps), len(qs))


import json
def save_tropical(ts, fname1):
    fname = fname1 + "-trop.txt"
    with nlw_open(fname) as f:
        prt("Saving tropical system...")
        # ts is a list of dicts with tuples of Ints as keys and Ints as values
        ts2 = [sorted(d.items()) for d in ts]
        s = "".join(["  " + json.dumps(d, separators=(',',':')) + ",\n" for d in ts2])
        if s:
            s = s[:-2] + "\n"
        f.write("[\n" + s + "]\n")


def load_tropical(fname1):
    fname = fname1 + "-trop.txt"
    try:
        with open(fname) as f:
            prt(end="Loading tropical system...", flush=True)
            try:
                ts2 = json.loads(f.read())
            except ValueError:
                prt(" error.")
                return None                             # error while decoding
            # ts is a list of dicts with tuples of Ints as keys and Ints as values
            ts = [{tuple(k): v for k,v in d} for d in ts2]
        prt(" done.")
        return ts
    except IOError:
        return None                                     # file could not be read


def load_jeff_system(dir):
    fname = dir + "newton.txt"
    ts = []
    cnt = 0
    try:
        with open(fname) as f:
            prt(end="Loading Newton polytopes... ", flush=True)
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    n = json.loads(l)
                    #prt(n)
                    d = {tuple([0] + k): 0 for k in n}
                    #prt(d)
                    ts.append(d)
                    prt(end="[{}]".format(cnt), flush=True)
                except ValueError:
                    prt(" error.")
                    return                              # error while decoding
                cnt += 1
                # ts is a list of dicts with tuples of Ints as keys and Ints as values
        prt(" done.")
        #prt(ts)
        return ts
    except IOError:
        return                                          # file could not be read


def rational_str(r):
    return str(r)
    #assert r.denominator() == 1
    #if r.denominator() == 1:
    #    return str(r.numer())
    #return str(r.numer()) + "/" + str(r.denominator())

def list_rational(l):
    return "[" + ",".join([rational_str(i) for i in l]) + "]"

def list_list(l, linestart=""):
    return "[" + (","+linestart).join([list_rational(i) for i in l]) + "]"

def disj_to_str(d, linestart=""):
    if d is None:
        return "None"
    return list_list(d.astype(int).tolist(), linestart="\n "+linestart)

def save_polyhedra(pts_bags, fname1, quiet=False):
    fname = fname1 + "-polyhedra.txt"
    with nlw_open(fname) as f:
        if not quiet:
            prt("Saving polyhedra...")
        s = "[\n"
        for b in pts_bags:
            s += '  ["{}",\n    '.format(b.name)
            s += disj_to_str(b.qdisj, linestart="    ") + ", [\n"
            s += ",\n".join(["    [" + list_list(p.equalities_list()) + ", " + list_list(p.inequalities_list()) + ((", " + str(p.oidx)) if "oidx" in dir(p) else ", -1") + "]" for p in b])
            s += "\n  ]],\n"
        if s:
            s = s[:-2] + "\n"
        s += "]\n"
        f.write(s)


def load_polyhedra(fname1):
    fname = fname1 + "-polyhedra.txt"
    from ptcut import PtsBag
    try:
        with open(fname) as f:
            prt(end="Loading polyhedra...", flush=True)
            try:
                bags2 = json.loads(f.read())
            except ValueError:
                prt(" error.")
                return None                             # error while decoding
            bags = []
            for bagnb,b2 in enumerate(bags2):
                b = []
                for cnt,p2 in enumerate(b2[2]):
                    p = phwrap(eqns=p2[0], ieqs=p2[1])
                    if p2[2] >= 0:
                        p.idx = p.oidx = p2[2]
                    else:
                        p.idx = cnt
                    p.combo = {bagnb: p.idx}
                    b.append(p)
                b = PtsBag(b)
                b.name = b2[0]
                b.qdisj = np.array(b2[1])
                assert b.name == "#{}".format(bagnb), "{} {}".format(b.name, bagnb)
                bags.append(b)
        prt(" done.")
        return bags
    except IOError:
        return None                                     # file could not be read


def time_prec():
    prt(end="\nMeasuring timer precision...", flush=True)
    # measure it
    runs = 100000
    val = [0] * runs
    for i in range(runs):
        val[i] = mytime()
    # calculate on it
    last = val[0]
    mindiff = 999
    maxdiff = 0
    sum = 0
    sum2 = 0
    cnt = 0
    for t in val[1:]:
        diff = (t - last) * 1e6
        if diff > 0:
            mindiff = min(mindiff, diff)
        maxdiff = max(maxdiff, diff)
        sum += diff
        sum2 += diff * diff
        cnt += 1
        last = t
    avg = sum / cnt
    dev = (sum2 / cnt - (sum/cnt) ** 2) ** 0.5
    prt("\nTimer precision: min={:.3f} avg={:.3f} max={:.3f} dev={:.3f} usec".format(mindiff, avg, maxdiff, dev), flush=True)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
