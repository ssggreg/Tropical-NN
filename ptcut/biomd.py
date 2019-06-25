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

import time, re
from util import mytime, fract, myround
from math import log
from prt import prt


def handle_params(ps):
    """
    Convert parameters from Params.txt file into dictionary.  If possible, the parameters have type int.

    >>> sorted(handle_params(["k1 = 1.2121", "k2 = 0.13889", "k3 = 1e-12", "k4 = 19000000", "k5 = 25", "k6 = 0.813", "k7 = 0.557", "k8 = 400000", "k9 = 100000"]).items())
    [('k1', 1.2121), ('k2', 0.13889), ('k3', 1e-12), ('k4', 19000000), ('k5', 25), ('k6', 0.813), ('k7', 0.557), ('k8', 400000), ('k9', 100000)]
    """
    d = {}
    for p in ps:
        p = p.split(";")[0]                             # remove comments
        if not p.strip():
            continue
        ar = p.split("=")
        assert len(ar) == 2
        f = float(ar[1])
        d[ar[0].strip()] = int(f) if int(f) == f else f
    return d


def logep(x, ep, scale=1):
    """
    >>> logep(3000,0.2)
    -5
    >>> logep(0.0001,0.2)
    6
    >>> logep(0.0001,1/5)
    6
    >>> logep(0.0001,fract(1,5))
    6
    >>> logep(100,10)
    2
    >>> logep(99,10,10)
    20
    >>> logep(80,10,10)
    19
    """
    return myround(log(x,ep) * scale)


def mysgn(x):
    """
    Simple self-made signum function.

    >>> mysgn(12)
    1
    >>> mysgn(-1009912)
    -1
    >>> mysgn(0.001232)
    1
    >>> mysgn(0)
    0
    """
    return -1 if x < 0 else 1 if x > 0 else 0


def flat_ordered_terms(f):
    """
    Return a flattened list of terms, i.e., even for nested Add's.

    >>> from sympy import sympify, Add
    >>> f = sympify("a+b")
    >>> list(flat_ordered_terms(f))
    [a, b]
    >>> g = sympify("b+c")
    >>> h = Add(f,g,evaluate=False)
    >>> list(flat_ordered_terms(h))
    [a, b, b, c]
    """
    from sympy import Add
    l = []
    for t in f.as_ordered_terms():
        if type(f) == Add:
            l.extend(flat_ordered_terms(t))
        else:
            l.append(t)
    return l


def handle_polysys(str_polys, params, ep, sumup, verbose=0, keep_coeff=False, paramwise=True, quiet=False, scale=1,
        cache=None, mul_denom=False):
    """
    ps is the polynomial system, one poly per line as string

    >>> d = handle_params(["k1 = 1.2121", "k2 = 0.13889", "k3 = 1e-12", "k4 = 19000000", "k5 = 25", "k6 = 0.813", "k7 = 0.557", "k8 = 400000", "k9 = 100000"])
    >>> s = ["-k4*x2*(-k8+x2+k9)+k5*(k8-x2-x4)", "k6*(k8-x2-x4)-k7*x4"]
    >>> [sorted(i.items()) for i in handle_polysys(s, d, 1.0/5, True, 0, True, False, True)[0]]
    [[((-18, 1, 0), 1), ((-10, 0, 0), 1), ((-10, 2, 0), -1), ((-2, 0, 1), -1)], [((-8, 0, 0), 1), ((0, 0, 1), -1), ((0, 1, 0), -1)]]
    """
    if quiet:
        verbose = False
    if paramwise:
        # Satya does parameter-wise rescaling.
        # To make this easy, round parameters according to ep before they are used.
        p2 = {}
        for k,v in params.items():
            p2[k] = ep ** (logep(v, ep, scale) / scale) if v > 0 else v
        params = p2
    if cache:
        vars, used_params, old_params, old_trop = cache
        changed_params = set([k for k in params if params[k] != old_params[k]])
        #prt(end="{}".format(changed_params))
    else:
        vars = set()
        used_params = []
    polys = []
    is_rat = False
    from sympy import sympify, Add, lcm, srepr
    bad_pattern = re.compile(r"\b(?:" + "|".join(["zoo", "nan"]) + r")\b")
    for p in str_polys:
        if cache and not (changed_params & used_params[len(polys)]):
            # we have a cache and there are no parameter changes from last round in this equation:
            # use old substituted formula.
            polys.append(None)                          # dummy; it's not used later anyway
            continue
        p = p.split(";")[0].strip()                     # remove comments
        if not p.split():                               # ignore empty lines
            continue
        if not quiet:
            print(end="[{}]".format(len(polys)))
        if verbose:
            prt("input: {}".format(p), flush=True)
        if bad_pattern.search(p):
            prt("System contains infinities, abort!")
            return None, cache
        f = sympify(p)                                  # convert string to sympy object
        if not cache:
            # save all symbols (as strings) used in that equation
            used_params.append(set([str(j) for j in f.free_symbols]))
        if verbose:
            prt("sympify into: {}".format(f), flush=True)
        f = f.expand()                                  # expand representation into terms
        if verbose:
            prt("expand into: {}".format(f), flush=True)
        # handle rational systems
        if mul_denom:
            l = []
            for t in flat_ordered_terms(f):              # list of terms
                _, denom = t.as_numer_denom()
                if denom.free_symbols:
                    if not is_rat:
                        is_rat = True
                        prt("System is rational.  Ignoring denominator.")
                    l.append(denom)
            mult = lcm(l)
            if mult != 1:
                # multiply list of terms with lcm of denoms
                prt("Multiplying equation {} with: {}".format(len(polys), mult))
                l = []
                for t in flat_ordered_terms(f):             # list of terms
                    t = (t*mult).simplify().expand()
                    l.append(t)
                f = Add(*l, evaluate=False)                 # must prevent evaluation to prevent cancelation and collection
                if verbose:
                    prt("after mul_denom: {}".format(f))
        # substitute "k" parameters
        if sumup:
            # if sumup == True, we substitute all parameters, which will re-evaluate the formula
            # and collect terms with matching variables.  ex: k1*x1 + k2*x1 becomes
            # 2*x1 + 3*x1 (intermediate) and 5*x1 in the end.  Even more dramatic, k1*x1 - k2*x1
            # will be == 0, if k1 == k2.
            # Satya doesn't want that, so we usually use sumup == False.
            f = f.subs(params).expand()
        else:
            # if sumup == False, we will substitute parameters term-by-term, thus preventing
            # them to be collected and possibly canceling themselves out.
            l = []
            for t in flat_ordered_terms(f):              # list of terms
                if not keep_coeff:
                    # if keep_coeff == False, then in a term like 2*k1*x1, the 2 will be ignored.
                    cm = t.as_coeff_mul()
                    if abs(cm[0]) != 1:
                        t /= abs(cm[0])                 # remove constant
                t = t.subs(params)                      # substitute "k" parameters
                l.append(t)
            f = Add(*l, evaluate=False)                 # must prevent evaluation to prevent cancelation and collection
        if verbose:
            prt("subs into: {}".format(f))
        polys.append(f)
        if f.free_symbols:
            vars |= f.free_symbols
        if verbose:
            prt()
        prt.flush_all()
    # sort variables alphabetically
    if not all([i[0] == "x" and i[1:].isdigit() for i in [str(j) for j in list(vars)]]):
        prt("all variables should be called xNNN, where NNN are digits")
    vars = sorted(list(vars), key=lambda x: int(str(x)[1:]))      # all variables still left
    if verbose:
        prt("variables: {}".format(vars))

    trop = []
    for p in polys:
        if cache and not (changed_params & used_params[len(trop)]):
            # we have a cache and there are no parameter changes from last round in this equation:
            # use old trop dict.
            trop.append(old_trop[len(trop)])
            continue
        d = {}
        for t in flat_ordered_terms(p):                  # list of terms
            _, denom = t.as_numer_denom()
            if denom.free_symbols:
                prt("{}System is rational, abort!".format("" if verbose else "\n"))
                return None, cache
            tv = [0] * (len(vars) + 1)
            sign = 1
            t = t.simplify()                            # make sure constants are correctly collected
            for v, exp in t.as_powers_dict().items():
                #prt("  {}**{}".format(v, exp))
                if v.is_constant():
                    tv[0] = int(logep(abs(v), ep, scale))
                    sign = mysgn(v)
                else:
                    i = vars.index(v) + 1               # index 0 is the constant
                    tv[i] = int(exp * scale)
            k = tuple(tv)                               # must convert to immutable tuple and int to remove sympy.core.numbers.One
            if k in d:
                if d[k] != sign:
                    # if we have two tropical points with different signs, save the sign as zero.
                    # this will disable opposite sign checking for this point.
                    d[k] = 0
            else:
                d[k] = sign
        trop.append(d)
        if verbose:
            prt("trop: {}, cons={}".format(d, is_conservation_constraint(d)))
        prt.flush_all()
    cache = set(vars), used_params, params, trop
    return trop, cache


def tropicalize_system(mod, mod_dir, ep, sumup=True, verbose=0, keep_coeff=False, paramwise=True, param_override={},
        scale=1, cache=None, mul_denom=False):
    # read parameters from file
    try:
        with open(mod_dir + "Params.txt") as f:
            str_params = f.readlines()
            params = handle_params(str_params)
    except FileNotFoundError:
        str_params = []
        params = {}
    # apply overrides, if any given.  those must be str/float pairs
    params.update(param_override.items())
    # read polynomial system from file
    with open(mod_dir + "Polynomial_system.txt") as f:
        str_polysys = f.readlines()
        start = mytime()
        r = handle_polysys(str_polysys, params, ep, sumup, verbose, keep_coeff, paramwise, scale=scale, cache=cache, mul_denom=mul_denom)
        total = mytime() - start
        print()
        prt("Tropicalization time: {:.3f} sec".format(total), flush=True, flushfile=True)
    return r


def read_grid_data(ss):
    """
    >>> read_grid_data("k1:100:200:10")
    [('k1', 100.0, 200.0, 10.0, False)]
    >>> read_grid_data("k10:1.5:2.5:0.1,k11:5:9:1")
    [('k10', 1.5, 2.5, 0.1, False), ('k11', 5.0, 9.0, 1.0, False)]
    >>> read_grid_data("")
    []
    >>> read_grid_data("k1:100:200:*10")
    [('k1', 100.0, 200.0, 10.0, True)]
    >>> read_grid_data("k1:100")
    [('k1', 100.0, 100.0, 1.0, False)]
    """
    r = []
    for s in ss.split(","):
        if s:
            a = s.split(":")
            a2 = a[2] if len(a) > 2 else a[1]
            if len(a) > 3:
                if a[3][0] == "*":
                    mult = True
                    a3 = a[3][1:]
                else:
                    mult = False
                    a3 = a[3]
            else:
                mult = False
                a3 = 1
            r.append((a[0], float(a[1]), float(a2), float(a3), mult))
    return r


def T_sample_grid(grid):
    x = sample_grid(grid)
    return [[j for j in i] for i in x]

def sample_grid(grid):
    """
    >>> T_sample_grid([("k1",1,5,1,False)])
    [[('k1', 1)], [('k1', 2)], [('k1', 3)], [('k1', 4)], [('k1', 5)]]
    >>> T_sample_grid([])
    [[]]
    >>> T_sample_grid([("k1",1,3,1,False), ("k2",6,7,1,False)])
    [[('k1', 1), ('k2', 6)], [('k1', 2), ('k2', 6)], [('k1', 3), ('k2', 6)], [('k1', 1), ('k2', 7)], [('k1', 2), ('k2', 7)], [('k1', 3), ('k2', 7)]]
    >>> T_sample_grid([("k1",1,8,2,True)])
    [[('k1', 1)], [('k1', 2)], [('k1', 4)], [('k1', 8)]]
    >>> T_sample_grid([("k1",1,8,2,True), ("k2",1,10,10,True)])
    [[('k1', 1), ('k2', 1)], [('k1', 2), ('k2', 1)], [('k1', 4), ('k2', 1)], [('k1', 8), ('k2', 1)], [('k1', 1), ('k2', 10)], [('k1', 2), ('k2', 10)], [('k1', 4), ('k2', 10)], [('k1', 8), ('k2', 10)]]
    """
    if not grid:
        yield []
    else:
        vars = [g[0] for g in grid]
        cnt = [g[1] for g in grid]                      # start values
        while True:
            yield list(zip(vars, cnt))                  # make sure it's copied, so the actual values are copied, not their references
            for i in range(len(cnt)):
                if grid[i][4]:
                    assert grid[i][3] != 1
                    cnt[i] *= grid[i][3]                # multiply by step size
                else:
                    assert grid[i][3] != 0
                    cnt[i] += grid[i][3]                # increment by step size
                if cnt[i] > grid[i][2]:                 # over upper limit?  (inclusive!)
                    cnt[i] = grid[i][1]
                else:
                    break                                # values ok, exit loop
            else:
                break                                    # all positions covered, exit outer loop


# less than 0.5 sec
biomd_fast = sorted([
    "BIOMD0000000108", "BIOMD0000000342",                # no solution for eps=1/5
    "BIOMD0000000005c_modified",
    "BIOMD0000000027_transfo", "BIOMD0000000027_transfo_qe", "BIOMD0000000029_transfo", "BIOMD0000000031_transfo",
    "BIOMD0000000035", "BIOMD0000000040", "BIOMD0000000072", "BIOMD0000000077", "BIOMD0000000101", "BIOMD0000000125", "BIOMD0000000150",
    "BIOMD0000000156", "BIOMD0000000159", "BIOMD0000000193", "BIOMD0000000194", "BIOMD0000000198", "BIOMD0000000199",
    "BIOMD0000000233", "BIOMD0000000257", "BIOMD0000000257c",
    "BIOMD0000000289", "BIOMD0000000361", "BIOMD0000000459", "BIOMD0000000460" ])

# more than 0.5 sec
biomd_slow = [ "BIOMD0000000001", "BIOMD0000000002", "BIOMD0000000009", "BIOMD0000000009p", "BIOMD0000000026", "BIOMD0000000026c",
    "BIOMD0000000028", "BIOMD0000000030", "BIOMD0000000038", "BIOMD0000000046",
    "BIOMD0000000080", "BIOMD0000000082", "BIOMD0000000102", "BIOMD0000000122", "BIOMD0000000123",
    "BIOMD0000000163", "BIOMD0000000226", "BIOMD0000000270",
    "BIOMD0000000287" ]

biomd_slowhull = [ "BIOMD0000000001", "BIOMD0000000002", "BIOMD0000000026", "BIOMD0000000028", "BIOMD0000000030", "BIOMD0000000038", "BIOMD0000000046",
    "BIOMD0000000080", "BIOMD0000000082", "BIOMD0000000123", "BIOMD0000000163", "BIOMD0000000270" ]

biomd_hard = [ "BIOMD0000000146_numer", "BIOMD0000000220p", "bluthgen0", "bluthgen1", "bluthgen2" ]
biomd_toohard = [ "BIOMD0000000019", "BIOMD0000000255", "BIOMD0000000335" ]

biomd_simple = sorted(biomd_fast + biomd_slow)
biomd_all = sorted(biomd_simple + biomd_hard)
biomd_easy = sorted(set(biomd_simple) - set(["BIOMD0000000102"]))


def load_known_solution(mod):
    try:
        from _biomdsoldb import biomd_sol_db
    except ImportError:
        return None
    try:
        return biomd_sol_db[mod.upper()]()
    except KeyError:
        return None


if __name__ == "__main__":
    import doctest
    doctest.testmod()
