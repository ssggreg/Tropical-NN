#! /usr/bin/env python3
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
    print('b')
    from sage.all_cmdline import *    # import sage library
    print('b')
    IS_SAGE = True
except ImportError:
    IS_SAGE = False

vsn = "3.3.3"
chull_f1 = 2
chull_f2 = 4
chull_f3 = 0.5
chull_f4 = 2

total_isect_time = 0
total_incl_time = 0
overall_isect_time = 0
overall_incl_time = 0

import os
import sys
import time
import numpy as np
import itertools
import random as randomx
import weakref
from util import *
from biomd import tropicalize_system, load_known_solution, biomd_simple, biomd_fast, biomd_slow, biomd_slowhull, biomd_easy, biomd_all, biomd_hard, read_grid_data, sample_grid
from prt import prt
from phwrapper import *
from bbox import *
from math import log, log10, ceil

try:
    from math import log2
except ImportError:
    # no log2 in Python 2.x
    log2 = lambda x: log(x, 2)


if not IS_SAGE:
    from vector import vector


def generate_polyhedra(points, bagnb, verbose=0, complex=False, nonewton=False):
    """
    From a dictionary of points with their sign (point is the key, sign the value)
    generate polyhedra.  The first coordinate of the keys is the absolute value,
    i.e. an entry of [1,7,3,4] represents the equality 1+7x1+3x2+4x3=0.

    Input: the input list of points.
        Example: points = {(0,6,0): -1, (0,3,1): 1, (0,3,0): -1, (0,1,2): 1}

    Each point is the tropicalization of a monomial.
    The value in the dictionary is +/-1 and contains the sign of that monomial.
    If the value is 0, then the opposite sign rule doesn't apply for that point.
    """

    # make it sorted, so it has a fixed, yet arbitrary order
    vs = sorted(points.keys())
    # all points must have the same number of coordinates
    assert all([len(p) == len(vs[0]) for p in points])

    if nonewton:
        newton_vertices = vs
    else:
        # build the Newton polytope of all points, but leaving off the parameter dimension
        newton = phwrap(vertices=vs)
        if verbose > 1:
            for p in vs:
                if not on_surface(newton, p):
                    prt("not on surface: {}".format(p))
        newton_vertices = [p for p in vs if on_surface(newton, p)]

    if verbose:
        prt("Points: {}".format(len(points)))
        assert nonewton or len(vs[0]) == newton.space_dim()
        prt("Dimension of the ambient space: {}".format(len(vs[0])))
        if not nonewton:
            prt("Dimension of the Newton polytope: {}".format(newton.dim()))
            prt(("Points on the border of the Newton polytope" + (" ({} total):" if verbose > 1 else ": {}")).format(len(newton_vertices)), flush=True)

        if verbose > 1:
            for idx,i in enumerate(newton_vertices):
                ii = tuple([int(j) for j in i])     # get rid of mpz (if existant)
                c = "+" if points[ii] > 0 else "-" if points[i] < 0 else "0"
                prt("    {}: {} {}".format(idx, ii, c))

            if not nonewton and verbose > 2:
                prt("Newton Hrep:")
                prt("    eqns={}".format(newton.equalities_list_i()))
                prt("    ieqs={}".format(newton.inequalities_list_i()))

    # cycle through all edges (1-faces) of the boundary of the Newton polytope
    l = []
    cnt = 0
    if verbose:
        prt("Border edges{}:".format("" if complex else " between points with opposing signs"), flush=True)
    for e in itertools.combinations(enumerate(newton_vertices), 2):
        v1 = vector(e[0][1])                            # one endpoint of the edge
        v2 = vector(e[1][1])                            # the other endpoint
        # must use tuple for dict access
        if (complex or points[e[0][1]] * points[e[1][1]] <= 0):
            if nonewton or adjacent_vertices(newton, v1, v2):
                # complex or opposing sign condition is met
                d = v1 - v2                                 # edge connecting both vertices
                # build list of inequalities:
                # v_1 = v_2 <= v_i, hence v_i - v_1 >= 0
                ie = []
                for v in newton_vertices:
                    vi = vector(v)
                    if vi != v1 and vi != v2:
                        ie.append(vi - v1)
                # eqns - list of equalities. Each line can be specified as any iterable container of
                #     base_ring elements. An entry equal to [1,7,3,4] represents the equality 1+7x1+3x2+4x3=0
                # ieqs - list of inequalities. Each line can be specified as any iterable container of
                #     base_ring elements. An entry equal to [1,7,3,4] represents the inequality 1+7x1+3x2+4x3>=0
                p = phwrap(eqns=[d], ieqs=ie)
                if verbose:
                    prt("    {}: {}-{}: {}: dim={}, compact={}".format("-" if p.is_empty() else cnt, e[0][0], e[1][0], p.Hrep(), p.dim(), p.is_compact()))
                if verbose > 1:
                    prt("    eq={}, ieqs={}".format(d, ie))
                if not p.is_empty():                        # exclude empty polyhedra
                    p.combo = {bagnb: cnt}
                    p.idx = p.oidx = cnt
                    cnt += 1
                    # save Newton polytope vertices that created this polyhedron
                    p.ij = set((e[0][0], e[1][0]))
                    l = insert_include0(l, p)
            elif verbose > 1:
                prt("Points {} and {} are not adjacent".format(v1, v2))
    l = PtsBag(l)
    # build qdisj matrix.  build always, so it can be saved
    l.qdisj = np.ones(shape=(len(l),len(l)), dtype=bool)
    np.fill_diagonal(l.qdisj, False)
    for i,j in itertools.combinations(range(len(l)), 2):
        if l[i].ij & l[j].ij:
            # possibly non-quasi-disjoint
            l.qdisj[i,j] = l.qdisj[j,i] = False
    # delete ij member, it's no longer needed
    for p in l:
        del p.ij
    if not l and len(points) == 1:
        prt("Warning: formula defines no polyhedron!  Ignored.")
        l = None
    if verbose and qdisjoint:
        if l:
            #with np.printoptions(threshold=1000000):
            # use numpy 1.15!!
            np.set_printoptions(threshold=1000000, formatter={'bool': '{:d}'.format})
            print("quasi-disjoint:")
            print(l.qdisj)
        prt()
    prt.flush_all()
    return l


def on_surface(p, v):
    """
    Check if a point is on the surface of the polyhedron.

    # a square in 2 dimensions can have interior points
    >>> p = phwrap(vertices=[[0,0], [2,0], [0,2], [2,2]])
    >>> on_surface(p, vector([0,0]))
    True
    >>> on_surface(p, vector([0,1]))
    True
    >>> on_surface(p, vector([1,1]))
    False
    >>> on_surface(p, vector([2,2]))
    True

    # a square in 3 dimensions has NO interior points
    >>> p = phwrap(vertices=[[0,0,0], [2,0,0], [0,2,0], [2,2,0]])
    >>> on_surface(p, vector([0,0,0]))
    True
    >>> on_surface(p, vector([0,1,0]))
    True
    >>> on_surface(p, vector([1,1,0]))
    True
    >>> on_surface(p, vector([2,2,0]))
    True

    # a line in 2 dimensions has NO interior points
    >>> p = phwrap(vertices=[[0,0], [2,0]])
    >>> on_surface(p, vector([0,0]))
    True
    >>> on_surface(p, vector([1,0]))
    True
    >>> on_surface(p, vector([2,0]))
    True

    # a line in 1 dimension has interior points
    >>> p = phwrap(vertices=[[0], [2]])
    >>> on_surface(p, vector([0]))
    True
    >>> on_surface(p, vector([1]))
    False
    >>> on_surface(p, vector([2]))
    True
    """
    if p.codim() > 0:
        return True
    # all the equalities must be true.
    for i in p.equalities_list():
        ii = vector(i[1:])
        if ii*v + i[0] != 0:
            return False
    # all the inequalities must be true.
    for i in p.inequalities_list():
        ii = vector(i[1:])
        if ii*v + i[0] < 0:
            return False
    # at least one inequality must be True as equality
    cnt = 0
    for i in p.inequalities_list():
        ii = vector(i[1:])
        if ii*v + i[0] == 0:
            cnt += 1
    return cnt > 0


def adjacent_vertices(p, v, w):
    """
    Check if two vertices (v and w) are adjacent,
    i.e. share the same facet of a polyhedron p.

    >>> p = phwrap(vertices=[[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    >>> adjacent_vertices(p, vector([-1, -1, -1]), vector([-1, -1, 1]))
    True
    >>> adjacent_vertices(p, vector([-1, -1, -1]), vector([-1, 1, 1]))
    False
    >>> adjacent_vertices(p, vector([-1, -1, -1]), vector([1, 1, 1]))
    False
    >>> adjacent_vertices(p, vector([-1, -1, -1]), vector([-1, -1, 0]))
    True

    >>> e = phwrap(eqns=[(1,2,4,6)])
    >>> q = p & e

    #>>> l = sorted(q.vertices_list())
    #>>> l
    #[[-1, -1, 5/6], [-1, 1, -1/2], [1/2, 1, -1], [1, -1, 1/6], [1, 3/4, -1]]
    >>> adjacent_vertices(q, vector([1, -1, 1/6]), vector([-1, -1, 5/6]))
    True
    >>> adjacent_vertices(q, vector([-1, 1, -1/2]), vector([1/2, 1, -1]))
    True

    #>>> adjacent_vertices(q, vector([1, 3/4, -1]), vector([-1, -1, 5/6]))
    #False
    """
    # every point on a polyhedron with codim > 0 is "on the surface"
    if p.codim() > 0:
        return True
    #return True
    if __debug__:
        # make sure we only get points that are on the surface of the polyhedron.
        assert v != w
        # all the equalities must be true for both.
        for i in p.equalities_list():
            ii = vector(i[1:])
            assert ii*v + i[0] == 0
            assert ii*w + i[0] == 0
        # all the inequalities must be true for both as well.
        for i in p.inequalities_list():
            ii = vector(i[1:])
            assert ii*v + i[0] >= 0
            assert ii*w + i[0] >= 0
    # at least one inequality must be shared as well.
    # check for equality, since we're only checking vertices on the hull anyway.
    cnt = 0
    for i in p.inequalities_list():
        ii = vector(i[1:])
        if ii*v + i[0] == 0 and ii*w + i[0] == 0:
            cnt += 1
    # since an edge has dimension 1, the two points have to share at least dim-1 facets.
    return cnt >= p.dim() - 1


class PtsBag(list):
    """
    A bag (set) of polyhedra.  Subclassed only so we can add another member.
    """
    def __init__(self, l):
        list.__init__(self, l)
        self.name = ""

def make_polyhedra(ts, skip_formulas=[], verbose=0, complex=False, nonewton=False):
    """
    From the tropicalization, create the polyhedra per bag.
    """
    assert ts
    start = mytime()
    pts_bags = []
    bagnb = 0
    for cnt,t in enumerate(ts):
        prt(end="[{}]".format(cnt), flush=True)
        if verbose:
            prt()
        if cnt in skip_formulas:
            prt(end="(skipped)", flush=True)
        else:
            p = generate_polyhedra(t, bagnb, verbose=verbose, complex=complex, nonewton=nonewton)
            if p != None:
                p.name = "#" + str(bagnb)
                pts_bags.append(p)
                bagnb += 1
    total = mytime() - start
    if not verbose:
        prt()
    prt("Creation time: {:.3f} sec".format(total), flush=True, flushfile=True)
    return pts_bags


def var_used_bool(b):
    """
    Measure which variables are used at all in a bag of polyhedra.

    >>> p = phwrap(eqns=[(0,0,0,4,0,6)], ieqs=[(0,0,2,5,0,4),(0,0,5,2,7,0),(0,0,0,2,0,5)])
    >>> var_used_bool([p])
    [False, True, True, True, True]
    """
    return [bool(i) for i in var_used_cnt(b)]


def var_used_cnt(b):
    """
    Count the number of uses of a specific variable in a bag of polyhedra.

    >>> p = phwrap(eqns=[(0,0,0,2,0,3)], ieqs=[(0,0,0,-1,0,0), (0,0,6,7,0,0), (0,0,5,2,7,0)])
    >>> var_used_cnt([p])
    [0, 2, 4, 1, 1]
    """
    uses = []
    for p in b:
        for h in p.Hrep():
            v = h.coeffs()     # this leaves out the absolute value
            if len(uses) < len(v):
                uses += [0] * (len(v) - len(uses))          # zero pad
            for cnt,coord in enumerate(v):
                uses[cnt] += coord != 0
    return uses


def likeness_15(b1, b2):
    """
    Predict size of intersection bag by v4 machine learning data.
    """
    l1, l2 = len(b1), len(b2)
    if l1 == 0 or l2 == 0:
        return inf
    u1 = var_used_bool(b1)
    #if not u1:
    #    return -inf
    #u2 = var_used_bool(b2)
    dim = len(u1)
    #uand = [i+j > 1 for i,j in zip(u1, u2)]
    #uor = [i+j > 0 for i,j in zip(u1, u2)]
    a = [
        log(l1,l1*l2),                                  #  3: b1: # polyhedra in bag
        avg_dim(b1) / dim,                              #  5: b1: avg object dim
        log(l2,l1*l2),                                  #  8: b2: # polyhedra in bag
        avg_dim(b2) / dim,                              # 10: b2: avg object dim
    ]
    pred = clf.predict([a])
    p = min(max(0, pred[0]), 1)  # safety
    return -p * l1 * l2


def likeness(b1, b2, like):
    """
    Calculate a measure of like-ness of two polyhedron bags.
    """
    if like == 15:
        return likeness_15(b1, b2)
    elif like == 14:
        return -len(b1)
    elif like == 0:
       return 0.0
    assert 0


def avg_dim(b):
    """
    Return the average dimension of a list of polyhedra.
    """
    return sum([p.dim() for p in b]) / float(len(b)) if len(b) > 0 else 0

def avg_planes(b):
    """
    Return the average number of defining hyperplanes of a list of polyhedra.
    """
    return sum([p.n_Hrep() for p in b]) / float(len(b)) if len(b) > 0 else 0

def avg_vrep(b):
    """
    Return the average number of defining vertices of a list of polyhedra.
    """
    return sum([p.n_Vrep() for p in b]) / float(len(b)) if len(b) > 0 else 0

def avg_vertices(b):
    """
    Return the average number of vertices of a list of polyhedra.
    """
    return sum([p.n_vertices() for p in b]) / float(len(b)) if len(b) > 0 else 0

def avg_rays(b):
    """
    Return the average number of rays of a list of polyhedra.
    """
    return sum([p.n_rays() for p in b]) / float(len(b)) if len(b) > 0 else 0

def avg_lines(b):
    """
    Return the average number of lines of a list of polyhedra.
    """
    return sum([p.n_lines() for p in b]) / float(len(b)) if len(b) > 0 else 0

def avg_compact(b):
    """
    Return the average number of is_compact() of a list of polyhedra.
    """
    return sum([p.is_compact() for p in b]) / float(len(b)) if len(b) > 0 else 0


include_tests = 0
intersections = 0

class NoSolution(Exception):
    pass

class OutOfTime(Exception):
    pass


def contains1(p, q, cnt, tcnt):
    global include_tests
    if do_times:
        if full_stat:
            try:
                pn = p.tmpname
            except AttributeError:
                pn = "?"
            try:
                qn = q.tmpname
            except AttributeError:
                qn = "?"
            prt(end="  incl {}/{}: ({}: dim={} hs={}) in ({}: dim={} hs={})".format(cnt, tcnt, qn, q.dim(), q.n_Hrep(), pn, p.dim(), p.n_Hrep()), flush=True)
        t = mytime()
    r = p.contains(q)
    if do_times:
        t = mytime() - t
        global total_incl_time
        total_incl_time += t
        if full_stat:
            prt(" = {}, time={:.6f}".format(int(r), t), flush=True, flushfile=True)
    include_tests += 1
    return r


def insert_include0(l, n):
    """
    Insert polyhedron n into list l, but check that n is not included
    in any of l's members and that n doesn't include any of l's members.
    If so, use the larger one.

    This is the simplest version of this function with no reordering of the list l.
    """
    # we're assuming that it is more likely that the new polyhedron is already included in
    # some polyhedron already in the list, so we check for that first.
    for i in l:
        # is n contained in i, i.e. something we already know?
        if i.contains(n):
            return l                                    # no need to continue
    # n is not included in any of l's polyhedra, so it has to be added to the list anyway.
    # see if n includes any of the already existing ones.
    l2 = []
    for i in l:
        # is i contained in n?  if so, don't copy it to new list.
        if not n.contains(i):
            # here, all inclusions are strict, otherwise they would have been found in pass 1.
            l2.append(i)                                # append i to new list
    l2.append(n)
    return l2


def insert_include(l, n, n2=None):
    """
    Insert polyhedron n into list l, but check that n is not included
    in any of l's members and that n doesn't include any of l's members.
    If so, use the larger one.
    If both are equal, keep the one with the smaller combo index.
    If n2 is specified and n was not found, add n2 instead.
    """
    #return l + [n]                                     # to switch off inclusion test
    strict_order = False
    total_cnt = (2 + strict_order) * len(l)
    cnt = 0
    # we're assuming that it is more likely that the new polyhedron is already included in
    # some polyhedron already in the list, so we check for that first.
    for idx,i in enumerate(l):
        # is n contained in i, i.e. something we already know?
        cnt += 1
        if contains1(i, n, cnt, total_cnt):
            cnt += strict_order
            if strict_order and contains1(n, i, cnt, total_cnt):
                # both are included in one another, i.e. they are equal.  keep the one with the lower combo index
                if combo_cmp(i.combo, n.combo) < 0:
                    l.insert(0, l.pop(idx))     # move to old i front of list
                else:
                    l.pop(idx)                          # remove old i with larger combo index
                    l.insert(0, n)                      # put new n to front of list
            else:
                # i is strictly larger than n
                l.insert(0, l.pop(idx))                 # move to front of list
            return l                                    # no need to continue
        else:
            cnt += strict_order
    # n is not included in any of l's polyhedra, so it has to be added to the list anyway.
    # see if n includes any of the already existing ones.
    l2 = []
    for i in l:
        # is i contained in n?  if so, don't copy it to new list.
        cnt += 1
        if not contains1(n, i, cnt, total_cnt):
            # here, all inclusions are strict, otherwise they would have been found in pass 1.
            l2.append(i)                                # append i to new list
    l2.append(n2 if n2 is not None else n)
    return l2


def contains2(p, q, cache, cnt, total_cnt):
    """
    Cached check if p contains q.

    both p and q are results of an intersection operation.
    call p = p1 \cap p2, q = q1 \cap q2.
    then q \subseteq p, i.e. p contains q, iff q \subseteq p1 AND q \subseteq p2.
    this reduces q*m inclusion checks to q+m checks.
    """
    # check parents[1] first, since the smaller bag is always the second of the parents
    # search for p1
    p1 = p.parents[1]()
    p1id = q.tmpname + p1.tmpname
    p2 = p.parents[0]()
    p2id = q.tmpname + p2.tmpname

    if cache.get(p2id, True) == False:
        return False

    try:
        r = cache[p1id]
    except KeyError:
        # it doesn't pay off to check if p1 is an educt to q.  contains is faster than my check.
        r = contains1(p1, q, cnt+1, total_cnt)
        cache[p1id] = r
    if r:
        # p1 contained q, let's check p2
        try:
            r = cache[p2id]
        except KeyError:
            r = contains1(p2, q, cnt+2, total_cnt)
            cache[p2id] = r
    return r


def insert_include2(l, n, cache):
    """
    Insert polyhedron n into list l, but check that n is not included
    in any of l's members and that n doesn't include any of l's members.
    If so, use the larger one.
    If both are equal, keep the one with the smaller combo index.
    """
    #return l + [n]                                     # to switch off inclusion test
    total_cnt = 4 * len(l)
    cnt = 0
    # we're assuming that it is more likely that the new polyhedron n is already included in
    # some polyhedron i already in the list l, so we check for that first.
    for idx,i in enumerate(l):
        # does i contain n, i.e. something we already know?
        if contains2(i, n, cache, cnt, total_cnt):
            # i is larger than or equal to n
            l.insert(0, l.pop(idx))                     # move to front of list
            return l                                    # no need to continue
        cnt += 2
    # n is not included in any of l's polyhedra, so it has to be added to the list.
    # see if n includes any of the already existing polyhedra.
    l2 = []
    for i in l:
        # does n contain i?  if so, don't copy it to new list.
        if not contains2(n, i, cache, cnt, total_cnt):
            # here, all inclusions are strict, otherwise they would have been found in pass 1.
            l2.append(i)                                # append i to new list
        cnt += 2
    # finally, add new n to list as well.
    l2.append(n)
    return l2


def insert_include3(l, n, cache, qdisj0, qdisj1):
    """
    Insert polyhedron n into list l, but check that n is not included
    in any of l's members and that n doesn't include any of l's members.
    If so, use the larger one.
    If both are equal, keep the one with the smaller combo index.
    """
    #return l + [n]                                     # to switch off inclusion test
    if not qdisjoint:
        return insert_include2(l, n, cache)

    def contains3a(p,q):
        if qdisj1[p.parents[1]().idx, q.parents[1]().idx] or qdisj0[p.parents[0]().idx, q.parents[0]().idx]:
            return False
        return contains2(p, q, cache, cnt, total_cnt)

    def contains3(p, q):
        # is q included in p?
        # this is sortof right
        if q.dim() > p.dim():
            # q is surely not included in p, since its dimension is higher than p's
            return False
        # both parents of p must be not quasi-disjoint with the respective parents of q for inclusion to be possible
        if q.dim() == p.dim() and (qdisj1[p.parents[1]().idx, q.parents[1]().idx] or qdisj0[p.parents[0]().idx, q.parents[0]().idx]):
            # dimensions are the same, but polyhedra are from quasi-disjoint ancestors, so they can't be included in one another
            return False
        return contains2(p, q, cache, cnt, total_cnt)

    def contains3_(p,q):
        r = contains3b(p,q)
        if r != contains3a(p,q):
            prt("\np =", p.Hrep(), "\nq =", q.Hrep(), "\ncap =", (p & q).Hrep())
        return r

    total_cnt = 4 * len(l)
    cnt = 0
    # we're assuming that it is more likely that the new polyhedron n is already included in
    # some polyhedron i already in the list l, so we check for that first.
    for idx,p in enumerate(l):
        # does p contain n, i.e. something we already know?
        if contains3(p, n):
            # p is larger than or equal to n
            l.insert(0, l.pop(idx))                     # move to front of list
            return l                                    # no need to continue
        cnt += 2
    # n is not included in any of l's polyhedra, so it has to be added to the list.
    # see if n includes any of the already existing polyhedra.
    l2 = []
    for p in l:
        # does n contain p?  if so, don't copy it to new list.
        if not contains3(n, p):
            # here, all inclusions are strict, otherwise they would have been found in pass 1.
            l2.append(p)                                # append p to new list
        cnt += 2
    # finally, add new n to list as well.
    l2.append(n)
    return l2


def myintersect(p, q, text=" "):
    if do_times:
        if full_stat:
            prt(end="  isect{}: (dim={} hs={}) x (dim={} hs={})".format(text, p.dim(), p.n_Hrep(), q.dim(), q.n_Hrep()), flush=True)
        t = mytime()
    r = p & q
    if do_times:
        t = mytime() - t
        global total_isect_time
        total_isect_time += t
        if full_stat:
            if r.is_empty():
                prt(" = (empty), time={:.6f}".format(t), flush=True, flushfile=True)
            else:
                prt(" = (dim={} hs={}), time={:.6f}".format(r.dim(), r.n_Hrep(), t), flush=True, flushfile=True)
            if t > 1:
                prt("___p={},_q={},_r={}\n___p.eq={}\n___p.ie={}\n___q.eq={}\n___q.ie={}\n___r.eq={}\n___r.ie={}".format(
                    comboname(p.combo), comboname(q.combo), comboname(r.combo),
                    p.equalities_list_i(), p.inequalities_list_i(),
                    q.equalities_list_i(), q.inequalities_list_i(),
                    r.equalities_list_i(), r.inequalities_list_i()).replace(" ", "").replace("_", " "), screen=False)
    r.combo = p.combo.copy()
    r.combo.update(q.combo)
    r.parents = weakref.ref(p), weakref.ref(q)                        # record parents for later inclusion test
    return r


def mk_stat_blob(b1, b2, l, empty_cnt, incl_cnt, drop_cnt):
    """
    Return statistics for both input and the output bag.

    They each contain:
        #polyhedra, #used variables
    """
    return None   # switched off for now

    u1 = var_used_bool(b1)
    if not u1:
        return None
    u2 = var_used_bool(b2)
    mls = "v4: "
    for i in (b1, b2, l):
        mls += "{} {} {:.2f} {:.2f} {:.2f} ".format(len(i), sum(var_used_bool(i)), avg_planes(i), avg_dim(i), avg_compact(i))
    uand = [i+j > 1 for i,j in zip(u1, u2)]
    uor = [i+j > 0 for i,j in zip(u1, u2)]
    mls += "{} {} {} {} {} {}".format(len(u1), empty_cnt, incl_cnt, drop_cnt, sum(uand), sum(uor))
    return mls


def full_random_product(a, b):
    lb = len(b)
    ab = len(a) * lb
    l = list(range(ab))
    randomx.shuffle(l)
    return tuple((a[x // lb], b[x % lb]) for x in l)


def disjoint_check(b):
    # check that no polyhedron is included in some other
    cnt, cnt_disj = 0, 0
    for p in b:
        for q in b:
            if p is not q:
                assert not p <= q
                cnt += 1
                cnt_disj += p.is_disjoint_from(q)
    print(b.name, cnt_disj, cnt)


def preprocess_isect(b1, b2):
    # go through all polyhedra and number them for later caching
    #disjoint_check(b1)
    for cnt,i in enumerate(b1):
        i.tmpname = "p" + str(cnt)
    #disjoint_check(b2)
    for cnt,i in enumerate(b2):
        i.tmpname = "q" + str(cnt)


def postprocess_isect(l, qdisj0, qdisj1):
    if qdisjoint:
        # enumerate new list
        for i in range(len(l)):
            l[i].idx = i
        # build new qdisj matrix
        # assume all are quasi-disjoint, except for diagonal
        l.qdisj = np.ones(shape=(len(l),len(l)), dtype=bool)
        np.fill_diagonal(l.qdisj, False)
        for p,q in itertools.combinations(l, 2):
            if not qdisj1[p.parents[1]().idx, q.parents[1]().idx] and not qdisj0[p.parents[0]().idx, q.parents[0]().idx]:
                l.qdisj[p.idx, q.idx] = l.qdisj[q.idx, p.idx] = False
    else:
        l.qdisj = None


def intersect_two(b1, b2, verbose=0, text="", like=9, endtime=inf, complete=False, filter=0):
    """
    Intersect all elements of p with all elements of q.
    Only save non-empty results and make sure that no element
    of the resulting list is included in any other element of it.
    """
    # make sure b1 is the larger of the two bags
    if len(b2) > len(b1):
        b1, b2 = b2, b1
    if verbose:
        stat_begin = "[{}({}, {})]: {} * {} = {}".format(text, b1.name, b2.name, len(b1), len(b2), len(b1)*len(b2))
        prt(end=stat_begin, flush=True)
        if full_stat:
            prt(":")
    # from the two bags, take all combinations of one polyhedron each and intersect those combinations.
    # if the intersection is empty, drop it.
    # if the intersection is contained in some other intersection we already have, drop it.
    # if the intersection is a superset of some other intersection we already have, drop that other intersection
    #    and add the new intersection.
    l = []                                              # new list of polyhedra
    empty_cnt = 0
    cnt = 0
    drop_cnt = 0
    tame = tame_it(2)
    filter_early = True and filter
    sts = status_print()
    stext = " "
    global include_tests
    global intersections
    if mytime() > endtime:
        raise OutOfTime
    # combine all of b1 with all of b2.  if filter > 0, pick random elements
    cache = {}
    preprocess_isect(b1, b2)
    for p,q in full_random_product(b1, b2) if filter_early else itertools.product(b1, b2):
        cnt += 1
        if full_stat:
            stext = " {}/{}".format(cnt, len(b1) * len(b2))
        elif tame.go():
            if mytime() > endtime:
                raise OutOfTime
            sts.print("({} => {})".format(cnt-1, len(l)))
        r = myintersect(p, q, text=stext)           # intersect
        r.tmpname = p.tmpname + "x" + q.tmpname
        intersections += 1
        if r.is_empty():                            # ignore if the intersection is empty
            empty_cnt += 1
            continue
        if complete and r.dim() != p.dim() - q.codim():
            drop_cnt += 1
            continue
        l = insert_include3(l, r, cache, b1.qdisj, b2.qdisj)
        if filter_early and len(l) >= filter:
            drop_cnt += len(b1) * len(b2) - cnt
            break
    if not full_stat:
        sts.print("")
    incl_cnt = len(b1)*len(b2) - len(l) - empty_cnt - drop_cnt
    if filter and len(l) > filter:
        # this will work if filter_early was not used.
        # sort bags by dimension & unboundedness.
        # the higher this value, the better the chances that it will "survive" the intersection process
        # shuffle first
        randomx.shuffle(l)
        # now sort
        l = sorted(l, key=lambda p: 2*p.dim() + (not p.is_compact()))
        # throw away "smallest" ones
        drop_cnt += len(l) - filter
        l = l[-filter:]
        assert len(l) == filter
    l = PtsBag(l)
    postprocess_isect(l, b1.qdisj, b2.qdisj)
    if verbose:
        s = stat_begin if full_stat else ""
        s += bag_status_line(l, empty_cnt, incl_cnt, drop_cnt)
        if like not in (0,14):
            lk = likeness(b1, b2, like)
            s += ", like={:.1f}".format(float(lk))
        if qdisjoint and len(l):
            s += ", qd={:.0%}".format(np.sum(l.qdisj / l.qdisj.size))
        prt(s, flush=True, flushfile=True)
    if len(l) == 0:
        raise NoSolution
    return l, mk_stat_blob(b1, b2, l, empty_cnt, incl_cnt, drop_cnt)


def highest_likeness(bags, like):
    minlike = []
    lmax = inf if like in minlike else -inf
    for i,b in enumerate(bags):
        for j,c in enumerate(bags):
            if like == 15 and j > 0:
                break
            if j >= i:
                break
            lk = likeness(b, c, like)
            if like in minlike:
                #lk = max(0.0, lk)
                #lk = min(1.0, lk)
                #prt("  {} {} {} {} {} {}".format(b.name, c.name, len(b), len(c), lk, lk * len(b) * len(c)))
                cmp = (1.0 - lk) * len(b) * len(c) < lmax
            else:
                cmp = lk > lmax
            if cmp:
                lmax = lk
                maxij = i, j
    j = maxij[0]
    i = maxij[1]
    assert j > i
    bj = bags[j]
    bags.pop(j)
    bi = bags[i]
    bags.pop(i)
    bags.insert(0, bi)
    bags.insert(1, bj)


# eqns - list of equalities. An entry equal to [1,7,3,4] represents the equality 1+7x1+3x2+4x3=0
# ieqs - list of inequalities. An entry equal to [1,7,3,4] represents the inequality 1+7x1+3x2+4x3>=0
def mkdict(l, is_eq, add_neg=False):
    """
    Convert a list of vectors (lists) to a dict (the 0-th coeff as value, the rest as key),
    optionally adding the negatives of those vectors as well since x = 2 implies x >= 2 as well as x <= 2).

    >>> sorted(mkdict([(1,2,3),(-4,-5,-6)], True).items())
    [((-5, -6), -4), ((2, 3), 1)]
    >>> sorted(mkdict([(1,2,3),(-4,-5,-6)], True, True).items())
    [((-5, -6), -4), ((-2, -3), -1), ((2, 3), 1), ((5, 6), 4)]
    """
    d = {}
    for v in l:
        v0 = v[0]                                       # the inhomogenous term
        vx = v[1:]                                      # the normal vector
        if is_eq:
            assert vx not in d or d[vx] == v0
            d[vx] = v0
        else:
            # if we have x-2 >= 0 (x >= 2) and x-4 >= 0 (x >= 4), then x-4 >= 0 (x >= 4) is true for both
            try:
                d[vx] = max(v0, d[vx])
            except KeyError:
                d[vx] = v0
        if add_neg:
            # negate, i.e. from 7x1+5x2 >= 2 generate 7x1+5x2 <= 2 as well
            vx = tuple([-i for i in vx])
            assert vx not in d or d[vx] == -v0          # add_neg must only be used for is_eq
            d[vx] = -v0
    return d


def dict_to_set(d):
    """
    Convert the normal vector/absolute dicts back to sets.

    >>> sorted(dict_to_set({(2,3):1, (5,6):4}))
    [(1, 2, 3), (4, 5, 6)]
    >>> sorted(dict_to_set({}))
    []
    """
    s = set()
    for k,v in d.items():
        s.add((v,) + k)
    return s


def join_dict(a, b):
    """
    >>> sorted(join_dict({1:2, 2:4}, {3:5, 4:9}).items())
    [(1, 2), (2, 4), (3, 5), (4, 9)]
    """
    d = a.copy()
    d.update(b)
    return d


def T_common_restrictions(b):
    """
    test companion function to cover directory randomness and version agnosticity.
    """
    r = common_restrictions(b)
    return [sorted([to_int(i) for i in s]) for s in r]

def common_restrictions(b):
    """
    Find the common restrictions for a list of polyhedra.

    >>> p = phwrap(eqns=[(1,2,3)], ieqs=[(4,5,6)])
    >>> q = phwrap(eqns=[(1,2,2)], ieqs=[(4,5,7)])
    >>> T_common_restrictions([p, q])
    [[], []]

    >>> p = phwrap(eqns=[], ieqs=[(4,5,6)])
    >>> q = phwrap(eqns=[], ieqs=[(4,5,6)])
    >>> T_common_restrictions([p, q])
    [[], [(4, 5, 6)]]

    >>> p = phwrap(eqns=[(1,2,3)], ieqs=[])
    >>> q = phwrap(eqns=[(2,2,3)], ieqs=[])
    >>> T_common_restrictions([p, q])
    [[], [(-1, -2, -3), (2, 2, 3)]]

    >>> p = phwrap(eqns=[], ieqs=[(1,2,3)])
    >>> q = phwrap(eqns=[], ieqs=[(2,2,3)])
    >>> T_common_restrictions([p, q])
    [[], [(2, 2, 3)]]
    """
    assert len(b) >= 2
    # initialize with first element of b
    comeq = mkdict(b[0].eqns_set, True)
    comie = join_dict(mkdict(b[0].ieqs_set, False), mkdict(b[0].eqns_set, True, True))

    # loop over the rest
    for p in b[1:]:
        eq = mkdict(p.eqns_set, True)
        ie = join_dict(mkdict(p.ieqs_set, False), mkdict(p.eqns_set, True, True))
        # check still common inequalities
        for k,v in comie.copy().items():
            if k not in ie:
                del comie[k]                            # it's not in both, i.e. delete in comie
            elif ie[k] != v:
                comie[k] = max(v, ie[k])               # if comie says x-2 >= 0 (x >= 2) and ie says x-4 >= 0 (x >= 4), then x-4 >= 0 (x >= 4) is true for both
        # check still common equalities
        for k,v in comeq.copy().items():
            if k not in eq:
                del comeq[k]                            # it's not in both, i.e. delete in comeq
            elif eq[k] != v:
                assert comie[k] >= v                    # this results from the adding of equations to comie
                comie[k] = max(comie[k], eq[k])         # if comeq says x-2 = 0 (x = 2) and eq says x-4 = 0 (x = 4), then x-2 >= 0 (x >= 2) is true for both
                del comeq[k]                            # no longer an equation
    # convert back
    return dict_to_set(comeq), dict_to_set(comie)


def mkset(l, add_neg=False):
    """
    Convert a list of vectors (lists) to a set,
    optionally adding the negatives of those vectors as well.

    >>> sorted(mkset([(1,2,3),(-4,-5,-6)]))
    [(-4, -5, -6), (1, 2, 3)]
    >>> sorted(mkset([(1,2,3),(-4,-5,-6)], True))
    [(-4, -5, -6), (-1, -2, -3), (1, 2, 3), (4, 5, 6)]
    """
    s = set()
    for v in l:
        s.add(v)
        if add_neg:
            s.add(tuple([-i for i in v]))
    return s


common_plane_time = 0
hull_time = 0

def common_planes(pts_bags, old_common, verbose=False, convexhull=True, endtime=inf, complete=False, bbox=False, common=9):
    """
    common == 0: don't do any common plane processing
    common == 1: only process bags with 1 polyhedron in it
    common == 2: like above, additionally search for common planes, but not half-spaces
    common == 9: do everything
    """
    start = mytime()
    global common_plane_time, hull_time
    #prt("Finding common planes... ")
    combi = prod([len(i) for i in pts_bags])
    old_len = len(pts_bags)
    global intersections
    oeq, oie, oldc = old_common
    worked = False
    loop = True
    debug_hull = False
    max_planes = 5000
    slow_build = False

    # loop will run as long as we find new constraints
    while loop:
        if mytime() > endtime:
            raise OutOfTime
        one_combo = {}
        one_cnt = 0

        # compute bounding boxes first
        bbox_c = None
        if bbox:
            a_bboxes = []
            for b in pts_bags:
                if "bbox" not in dir(b):
                    # bag bbox not cached
                    assert len(b)
                    # get the bounding box that contains all polyhedra in this bag
                    b_bboxes = []
                    for p in b:
                        if "bbox" not in dir(p):
                            # polyhedron bbox not cached
                            p.make_v()
                            p.bbox = p.get_bounding_box()
                        b_bboxes.append(p.bbox)
                    b.bbox = bbox_union(b_bboxes)
                a_bboxes.append(b.bbox)
            # build the intersection of all per-bag bounding boxes
            mins, maxs = bbox_intersection(a_bboxes)
            # build polyhedron
            bbox_c = bbox_to_polyhedron(mins, maxs)

        if common:
            common_eq = set()
            common_ie = set()
            hull_eq = set()
            hull_ie = set()
            one_eq = set()
            one_ie = set()
            for b in pts_bags[:]:
                ## make sure b.hull exists
                #if "hull" not in b.__dict__:
                #    b.hull = False
                if len(b) == 1:
                    # bags with just one polyhedron
                    one_eq.update(mkset(b[0].equalities_tuple_list()))
                    one_ie.update(mkset(b[0].inequalities_tuple_list()))
                    one_cnt += 1
                    one_combo.update(b[0].combo)
                    pts_bags.remove(b)      # no longer needed
                elif b and common > 1:
                    # find common (in)equalities between all polyhedra.
                    beq, bie = common_restrictions(b)
                    if convexhull:
                        vertices = set()
                        rays = set()
                        lines = set()
                        for p in b:
                            vertices.update(mkset(p.vertices_tuple_list()))
                            rays.update(mkset(p.rays_tuple_list()))
                            lines.update(mkset(p.lines_tuple_list()))
                    common_eq.update(beq)
                    common_ie.update(bie)
                    # build convex hull of all polyhedra of this bag
                    b.hull = False
                    if convexhull and len(vertices)+len(rays)+len(lines) <= max(avg_dim(b) * chull_f1, avg_planes(b) * chull_f3):
                        if debug_hull:
                            prt(end="[hull {}v".format(len(vertices)+len(rays)+len(lines)), flush=True)
                        start_hull = mytime()
                        hull = phwrap(vertices=vertices, rays=rays, lines=lines)
                        hull_time += mytime() - start_hull
                        assert not hull.is_empty()
                        hull_eq.update(mkset(hull.equalities_tuple_list()))
                        hull_ie.update(mkset(hull.inequalities_tuple_list()))
                        b.hull = True
                        if debug_hull:
                            prt(end=" {}h]".format(hull.n_Hrep()), flush=True)

            # remove already known planes, but not from one_*, since they are filtered out later
            common_eq -= oeq
            hull_eq -= oeq
            common_ie -= oie
            hull_ie -= oie

            # make sure we're not using too many planes
            # we're only limiting the number of inequalities.
            n = max_planes
            if convexhull:
                n = min(n, int(avg_dim(b) * chull_f2))
            xie = common_ie | hull_ie
            if len(xie) > n:
                if debug_hull:
                    prt(end="[sample {} {}]".format(n, len(xie)), flush=True)
                xie = set(randomx.sample(xie, n))
            # any new planes?
            eq = one_eq | common_eq | hull_eq
            ie = one_ie
            if common > 2:
                ie |= xie
            if not eq - oeq and not ie - oie:
                break
            oeq |= eq
            oie |= ie
            # build a polyhedron of all planes
            if debug_hull:
                prt(end="[cut {}h {} {} {} {} {} {}".format(len(eq)+len(ie), len(one_eq), len(common_eq), len(hull_eq), len(one_ie), len(common_ie), len(hull_ie)), flush=True)
                # prt("eq={}\nie={}".format(eq, ie))
            assert eq | ie
            if slow_build:
                c = bbox_c  # may be None
                for cnt,i in enumerate(eq):
                    s = "".join([(" {:+}*x{}".format(j, c2) if j else "") for c2,j in enumerate(i[1:])])
                    s = "{}{} == 0".format(i[0], s)
                    prt("Adding equality {}/{}: {}".format(cnt, len(eq), s), flush=True, flushfile=True)
                    c1 = phwrap(eqns=[i])
                    if c is None:
                        c = c1
                    else:
                        c &= c1
                for cnt,i in enumerate(ie):
                    s = "".join([(" {:+}*x{}".format(j, c2) if j else "") for c2,j in enumerate(i[1:])])
                    s = "{}{} >= 0".format(i[0], s)
                    prt("Adding inequality {}/{}: {}".format(cnt, len(ie), s), flush=True, flushfile=True)
                    c1 = phwrap(ieqs=[i])
                    if c is None:
                        c = c1
                    else:
                        c &= c1
            else:
                if bbox_c:
                    c = phwrap(eqns=list(eq) + bbox_c.equalities_list(), ieqs=list(ie) + bbox_c.inequalities_list())
                else:
                    c = phwrap(eqns=eq, ieqs=ie)
            if debug_hull:
                if phwrap().has_v:
                    prt(end=" {}v]".format(c.n_Vrep()), flush=True)
                else:
                    prt(end="]", flush=True)
        else:
            c = bbox_c

        if c.is_empty():
            # empty polyhedron, i.e. no solution
            common_plane_time += mytime() - start
            raise NoSolution
        if convexhull:
            hull_ok = True
            # test against polyhedron without hull
            if debug_hull:
                prt(end="[recut {}h".format(len(eq)+len(ie)), flush=True)
            c2 = phwrap(eqns=eq, ieqs=common_ie | one_ie)
            if debug_hull:
                prt(end=" {}v]".format(c.n_Vrep()), flush=True)
            if c.n_Vrep() > c2.n_Vrep() * chull_f4:
                hull_ok = False
                c = c2
                if debug_hull:
                    prt(end="[not used]", flush=True)
        c.combo = one_combo
        c.idx = 0
        # is new c any more restrictive?
        if c.n_Hrep() == 0 or oldc is not None and c.contains(oldc):
            break
        oldc = c

        loop = False
        worked = True
        bags = []
        cnt = 1
        if full_stat:
            verbose = True
        prt(" Applying {} (codim={}, hs={}{})...".format("common planes" if common else "bounding boxes",
            c.codim(), c.n_Hrep(), ", vs={}".format(c.n_Vrep()) if phwrap().has_v else ""), flush=True)
        stext = " "
        once = True
        for b in pts_bags:
            assert len(b) != 1
            empty_cnt = 0
            stat_begin = " [{}/{} ({})]: {}".format(cnt, len(pts_bags), b.name, len(b))
            prt(end=stat_begin, flush=True, screen=verbose)
            if full_stat:
                prt(":", screen=verbose)
            cnt += 1
            b2 = []
            pcnt = 0
            obag = b.name[0] == "#"
            if obag:
                start_bname = [p.oidx for p in b]
            tame = tame_it(2)
            sts = status_print()
            for p in b:
                pcnt += 1
                if full_stat:
                    stext = " {}/{}".format(cnt, len(b))
                elif verbose and tame.go():
                    if mytime() > endtime:
                        raise OutOfTime
                    sts.print("({} => {})".format(pcnt - 1, len(b2)))
                # strangely, it's faster to use r = p & c compared to r &= c (17.5s vs. 18.1s on bluthgen0)
                r = myintersect(p, c, text=stext)
                intersections += 1
                if r.is_empty():
                    empty_cnt += 1
                    continue
                if obag:
                    r.oidx = p.oidx
                b2 = insert_include(b2, r, None if once else p)
            if not full_stat and verbose:
                sts.print("")
            if obag:
                end_bname = [p.oidx for p in b2]
                extra_str = ",".join(["{}.{}".format(b.name[1:], i) for i in set(start_bname) - set(end_bname)])
            #once = False
            b2 = PtsBag(b2)
            b2.name = b.name
            postprocess_isect(b2, b.qdisj, np.zeros(shape=(1,1), dtype=bool))
            bags.append(b2)
            loop |= len(b2) == 1                    # loop if resulting bag has only one polyhedron
            incl_cnt = len(b) - len(b2) - empty_cnt
            s = stat_begin if full_stat else ""
            s += bag_status_line(b2, empty_cnt, incl_cnt)
            prt(s, screen=verbose)
            if obag and extra_str:
                prt("  Removed: {}".format(extra_str), screen=verbose)
            prt.flush_all()
            if len(b2) == 0:
                common_plane_time += mytime() - start
                raise NoSolution
        pts_bags = bags
        if not pts_bags:
            break
    # make sure one_combo is not lost if there are no other bags.
    if not pts_bags:
        # if no bags left, return the common intersection c
        pts_bags = [PtsBag([c])]
        pts_bags[0].name = "c"
    for p in pts_bags[0]:
        p.combo.update(one_combo)
    if worked:
        prd = prod([len(i) for i in pts_bags])
        combi = combi // prd if combi / prd >= 100 else combi / prd
        prt(" Savings: factor {:,}{}, {} formulas".format(combi,
         (" (10^{})".format(myround(log10(combi))) if combi > 10 else ""), old_len - len(pts_bags)), flush=True)
    common_plane_time += mytime() - start
    return pts_bags, (oeq, oie, oldc)


def work_estimate(bags):
    """
    #>>> work_estimate([[1,2,3], [1,2]])
    #6
    #>>> work_estimate([[1,2,3], [1,2], [1,2,3,4]])
    #30
    """
    return prod(len(i) for i in bags)

    lens = [len(i) for i in bags]
    #print(lens)
    cnt = 0
    while len(lens) > 1:
        lens = [lens[0] * lens[1]] + lens[2:]
        cnt += lens[0]
    #print(cnt)
    return cnt


def intersect_all(pts_bags, verbose=0, algo=0, like=9, common=9, sorting=0, resort=False, convexhull=True, fname1=None,
        max_run_time=inf, bag_order=[], complete=False, filter=0, bbox=False):
    """
    Calculate the intersections of one polyhedron per bag.

    Input: 'pts_bags' is a list of lists of polyhedra ("bags of polyhedra").

    Output: a list of polyhedra.

    Take all combinations of one polyhedron per "bag".
    Intersect the polyhedra of each combination.
    If, in the result, one polyhedron is included in another, drop the included one from the solution set.
    """
    stats = {"max_pt": 0, "maxin_pt": 0, "aborted": False}
    if not pts_bags:                                    # no input
        return [], stats
    start_combis = combis = work_estimate(pts_bags)
    old_common = set(), set(), None
    dim = -2
    for i in pts_bags:
        if i:
            dim = i[0].space_dim()
            break
    if verbose:
        if dim != -2:
            prt("Dimension: {}".format(dim))
            prt("Formulas: {}".format(len(pts_bags)))
            prt(end="Order:")
            for i in pts_bags:
                prt(end=" {}".format(i.name[1:]))
            prt()
            prt(end="Possible combinations: ")
            for cnt,i in enumerate(pts_bags):
                prt(end="{}{} ".format("* " if cnt > 0 else "", len(i)))
            prt("= {:,}{}".format(combis, (" (10^{})".format(myround(log10(combis)))) if combis > 0 else ""), flush=True)
    if complete and len(pts_bags) > dim:
        prt("*** Warning: --complete used on overdetermined system!  Solution will be empty!")
    if fname1:
        fname1 += "-temp"
    global intersections, include_tests, common_plane_time, hull_time, total_isect_time, total_incl_time
    global overall_isect_time, overall_incl_time
    intersections = 0
    include_tests = 0
    common_plane_time = 0
    hull_time = 0
    total_isect_time = 0
    total_incl_time = 0
    start = mytime()
    ftime = 0
    endtime = mytime() + max_run_time

    try:
        if algo == 2:
            prt("Looking for best start bag:"),
            lmax = -1
            for i,b in enumerate(pts_bags):
                for j,c in enumerate(pts_bags):
                    if j > i:
                        lk = likeness(b, c, like)
                        if lk > lmax:
                            lmax = lk
                            maxij = (i, j)
            m = maxij[0]
            prt(pts_bags[m].name)
            pts_bags = [pts_bags[m]] + pts_bags[:m] + pts_bags[m+1:]
        run = 1
        runs = len(pts_bags) - 1
        maxrun = ceil(log2(runs)) if runs > 1 else 0
        work_cnt = 1
        while len(pts_bags) > 1:
            if common or bbox:
                # find common planes and apply them to all bags
                old = len(pts_bags)
                pts_bags, old_common = common_planes(pts_bags, old_common, verbose=verbose > 1, convexhull=convexhull and run == 1, endtime=endtime, complete=complete, bbox=bbox, common=common)
                if len(pts_bags) <= 1:
                    break
                runs -= old - len(pts_bags)
                # if resorting is on, keep it sorted in every step
                if resort:  # why did I add " or run == 1" ?  cl, 11-12-2018
                    if sorting == 1:
                        pts_bags = sorted(pts_bags, key=lambda x: len(x))
                    elif sorting == 2:
                        pts_bags = sorted(pts_bags, key=lambda x: -len(x))

            if algo == 0:
                # standard breadth first
                l, mls = intersect_two(pts_bags[0], pts_bags[1], verbose, text="{}/{} ".format(run, runs), like=like, endtime=endtime, complete=complete, filter=filter)
                pts_bags = [l] + pts_bags[2:]
            elif algo == 1:
                # join and conquer
                i = 0
                bags2 = []
                while i < len(pts_bags) - 1:
                    b1 = pts_bags[i]
                    b2 = pts_bags[i+1]
                    stats["maxin_pt"] = max(stats["maxin_pt"], len(pts_bags[0]) * len(pts_bags[1]))
                    l, mls = intersect_two(b1, b2, verbose, like=like, text="{}/{}: ".format(run, maxrun), endtime=endtime, complete=complete, filter=filter)
                    l.name = "w{}".format(work_cnt)
                    work_cnt += 1
                    bags2.append(l)
                    i += 2
                # odd number of bags?
                if i < len(pts_bags):
                    bags2.append(pts_bags[i])
                pts_bags = bags2
            elif algo == 2:
                # find the next one that has the highest likeness
                lmax = -1
                nhmin = 2**63
                if like == 9 and True:
                    gavg = combis ** (1.0 / len(pts_bags))
                    if gavg > len(pts_bags[0]):
                        bags2 = [i for i in pts_bags[1:] if len(i) >= gavg]
                        bags1 = [i for i in pts_bags[1:] if not len(i) >= gavg]
                    else:
                        bags2 = [i for i in pts_bags[1:] if len(i) <= gavg]
                        bags1 = [i for i in pts_bags[1:] if not len(i) <= gavg]
                    #prt(gavg, [len(p) for p in bags2])
                else:
                    bags1 = []
                    bags2 = pts_bags[1:]
                for b in bags2[:]:
                    lk = likeness(pts_bags[0], b, like)
                    nh = avg_planes(b)
                    bags2 = bags2[1:]
                    if lk > lmax or (like == 3 and lk == lmax and nh < nhmin):
                        lmax = lk
                        nhmin = nh
                        b1max = bags1[:]
                        b2max = bags2[:]
                        bmax = b
                    bags1.append(b)
                l, mls = intersect_two(pts_bags[0], bmax, verbose, like=like, text="{}/{} ".format(run, runs), endtime=endtime, complete=complete, filter=filter)
                l.name = "w"
                pts_bags = [l] + b1max + b2max
            elif algo == 3:
                # if a bag order was specified, the specified bags have already been sorted to the front.
                # if we encounter an unlisted bag it means the ordered bags list has been consumed and
                # the automatism should take over.
                if not int(pts_bags[1].name[1:]) in bag_order:
                    # find the next pair that has the highest likeness
                    highest_likeness(pts_bags, like)         # sort the two with the highest likeness to the front
                stats["maxin_pt"] = max(stats["maxin_pt"], len(pts_bags[0]) * len(pts_bags[1]))
                l, mls = intersect_two(pts_bags[0], pts_bags[1], verbose, like=like, text="{}/{} ".format(run, runs), endtime=endtime, complete=complete, filter=filter)
                l.name = "w{}".format(work_cnt)
                work_cnt += 1
                pts_bags = [l] + pts_bags[2:]
            run += 1
            stats["max_pt"] = max(stats["max_pt"], len(pts_bags[0]))
            if fname1:
                save_polyhedra(pts_bags, fname1, quiet=True)
            if mls:
                print(mls, file=mls_file)
                mls_file.flush()
            combis = work_estimate(pts_bags)
        assert len(pts_bags) == 1

    except NoSolution:
        pts_bags = [PtsBag([])]
    except OutOfTime:
        prt("\n****** Out of time - Computation aborted! ******")
        stats["aborted"] = True
        pts_bags = [PtsBag([])]
    except KeyboardInterrupt:
        prt("\n****** Ctrl-C - Computation aborted! ******")
        stats["aborted"] = 2
    except MemoryError:
        prt("\n****** Out of memory - Computation aborted! ******")
        stats["aborted"] = True

    stats["max_pt"] = max(stats["max_pt"], len(pts_bags[0]))
    total = mytime() - start - ftime
    overall_isect_time += total_isect_time
    overall_incl_time += total_incl_time

    prt("Total time: {:.3f} sec, total intersections: {:,}, total inclusion tests: {:,}{}".format(
        total, intersections, include_tests, " (aborted)" if stats["aborted"] else ""))
    if do_times:
        prt("Total intersection time: {:.3f} sec, total inclusion time: {:.3f} sec".format(total_isect_time, total_incl_time))
    if common:
        prt("Common plane time: {:.3f} sec{}".format(common_plane_time, ", convex hull time: {:.3f} sec".format(hull_time) if convexhull else ""))
    if stats["aborted"]:
        #prt("Completed: {:.2f} %".format(100 - 100 * combis / start_combis))
        prt("Completed: {:.2f} log%, 10^{:.2f} combinations".format(100 - 100 * log10(combis) / log10(start_combis), log10(start_combis / combis)))
    prt.flush_all()

    if fname1:
        try:
            os.remove(fname1 + "-polyhedra.txt")
        except OSError:
            pass
    return pts_bags[0], stats


def bag_status_line(b, empty_cnt, incl_cnt, drop_cnt=0):
    s = " => {}".format(len(b))
    ss = []
    if empty_cnt > 0:
        ss.append("{} empty".format(empty_cnt))
    if incl_cnt > 0:
        ss.append("{} incl".format(incl_cnt))
    if drop_cnt > 0:
        ss.append("{} drop".format(drop_cnt))
    s += " ({})".format(", ".join(ss) if ss else "-")
    s += ", dim={:.1f}, hs={:.1f}".format(avg_dim(b), avg_planes(b))
    # used={}, sum(var_used_bool(b))
    #" (hull)" if b.hull and hull_ok else
    if phwrap().has_v:
        s += ", vs={:.1f}".format(avg_vrep(b))
    return s


def status(time, file=sys.stdout):
    print("\n" + "-" * 70, file=file)
    if IS_SAGE:
        print(version(), file=file)
    else:
        print(end="Plain ", file=file)
    s = "Python {}{}\nnumpy {}".format(sys.version, "" if __debug__ else ", (no debug)", np.__version__)
    try:
        import gmpy2
        s += ", gmpy2 {}".format(gmpy2.version())
    except ImportError:
        pass
    print(s, file=file)
    #import fuse
    print("This is ptcut v{}".format(vsn), file=file)
    print("It is now {}".format(time), file=file)
    print("command line: {}".format(" ".join(sys.argv)), file=file)


def T_get_ep(i):
    r = get_ep(i)
    return str(r[0]), r[1]

def get_ep(i):
    """
    Read the value of 1/ep.  Can be one of four formats:
    1. "<int>": ep = 1/<int>.
    2. "p<idx>": ep is the reciprocal of the smallest prime number with <idx> many digits.
    3. "c<idx>": ep is the reciprocal of the smallest prime number with <idx> many digits + 1 (thus it's surely composite).
    4. "<float>": ep 1/<float>, where <float> is acurate (it's expressed as a rational from string representation).

    >>> T_get_ep("5")
    ('1/5', '5')
    >>> T_get_ep("01001")
    ('1/1001', '1001')
    >>> T_get_ep("p3")
    ('1/1009', 'p3')
    >>> T_get_ep("c4")
    ('1/10008', 'c4')
    >>> T_get_ep("e2")
    ('1/100', 'e2')
    >>> T_get_ep("1.1")
    ('10/11', '11d10')
    """
    if i[0] in "pce" and i[1:].isdigit():
        idx = int(i[1:])
        if i[0] == "e":
            rep = 10 ** idx
        else:
            try:
                rep = make_prime(idx) + (1 if i[0] == "c" else 0)
            except IndexError:
                print("No prime with {} digits found".format(idx))
                sys.exit(1)
        i = i[0] + str(idx)                             # make it canonical
        ret = fract(1, rep), i
    else:
        try:
            rep = int(i)
            ret = fract(1, rep), str(rep)               # make it canonical
        except ValueError:
            from fractions import Fraction
            rep = Fraction(i)                           # construct from string to avoid rounding problems
            rep = fract(rep.numerator, rep.denominator) # convert to SAGE, possibly
            n = "{}d{}".format(rep.numer(), rep.denom())
            ret = 1 / rep, n
    assert 0 < ret[0] < 1
    return ret


default_epname = "5"

def main():
    total_time = mytime()
    from datetime import datetime
    boot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status(boot_time)

    print_out = False
    sorting = 1
    verbose = 0
    eps = []
    models = []
    algo = 3
    like = 14
    collect = 0
    skip_formula = []
    common = 9
    sumup = False
    rnd = 0
    force = False
    keep_coeff = False
    resort = False
    paramwise = True
    test = False
    convexhull = True
    grid_data = []
    _log_file_name = None
    complex = False
    save_solutions = True
    bag_order = []
    fusion = False
    save_solutions_to_log = False
    con_type = 0
    connected_components = False
    multiple_lpfiles = False
    jeff = False
    multi_log = False
    _runs = 1
    max_run_time = inf
    log_append = False
    work_dir = ""
    db_dir = "db"
    complete = False
    comp_sol = True
    lifting = True
    filter = 0
    global full_stat, do_times
    full_stat = False
    do_times = False
    set_phwrap(PhWrapPplCPolyhedron)
    import codecs
    random_seed = int(codecs.encode(os.urandom(8), "hex"), 16)
    bounding_box = False
    global qdisjoint
    qdisjoint = False
    remove_ph = []
    nonewton = False
    mul_denom = False

    for i in sys.argv[1:]:
        if collect:
            if collect == 1:
                skip_formula.extend([int(j) for j in i.split(",")])
            elif collect == 2:
                global chull_f1
                chull_f1 = bestnum(i)
            elif collect == 3:
                global chull_f2
                chull_f2 = bestnum(i)
            elif collect == 4:
                for j in i.split(","):
                    eps.append(get_ep(j))
            elif collect == 5:
                global chull_f3
                chull_f3 = bestnum(i)
            elif collect == 6:
                global chull_f4
                chull_f4 = bestnum(i)
            elif collect == 7:
                grid_data.extend(read_grid_data(i))
            elif collect == 8:
                rnd = int(i)
            elif collect == 9:
                _log_file_name = i
                multi_log = True
            elif collect == 10:
                bag_order.extend([int(j) for j in i.replace(":", ",").split(",")])
            elif collect == 11:
                con_type = int(i)
            elif collect == 12:
                _runs = int(i)
            elif collect == 13:
                max_run_time = float(i)
            elif collect == 14:
                work_dir = i
            elif collect == 15:
                db_dir = i
            elif collect == 16:
                filter = int(eval(i))
            elif collect == 17:
                random_seed = int(i, 36)
            elif collect == 18:
                models = [x for x in models if x not in i.split(",")]
            elif collect == 19:
                remove_ph.extend([j for j in i.split(",")])
            collect = 0
        elif i.startswith("-"):
            if i == "-t":
                _runs = 2**63
            elif i == "-p":
                print_out = True
            elif i == "--sortup":
                sorting = 1
            elif i == "--sortdown":
                sorting = 2
            elif i == "--shuffle":
                sorting = 3
            elif i == "--noshuffle":
                sorting = 0
            elif i.startswith("-a"):
                algo = int(i[2:])
            elif i.startswith("-l"):
                like = int(i[2:])
            elif i == "--bp":
                set_phwrap(PhWrapPolyhedronPPL)
            elif i == "--bc":
                set_phwrap(PhWrapPplCPolyhedron)
            elif i == "--bd":
                set_phwrap(PhWrapPolyhedronCDD)
            elif i == "--bn":
                set_phwrap(PhWrapPolyhedronNormaliz)
            elif i == "--bf":
                set_phwrap(PhWrapPolyhedronField)
            elif i == "--bm":
                set_phwrap(PhWrapPolyhedronPolymake)
            elif i.startswith("-v"):
                for j in i[1:]:
                    if j != 'v':
                        break
                    verbose += 1
            elif i == "--verbose":
                verbose += 1
            elif i == "--simple":
                models.extend(biomd_simple)
            elif i == "--easy":
                models.extend(biomd_easy)
            elif i == "--fast":
                models.extend(biomd_fast)
            elif i == "--slow":
                models.extend(biomd_slow)
            elif i == "--slowhull":
                models.extend(biomd_slowhull)
            elif i == "--hard":
                models.extend(biomd_hard)
            elif i == "--all":
                models.extend(biomd_all)
            elif i == "--no":
                collect = 18
            elif i == "--skip":
                collect = 1
            elif i == "--common":
                common = 9
            elif i == "--common1" or i == "--one":
                common = 1
            elif i == "--common2":
                common = 2
            elif i == "--nocommon" or i == "--nc":
                common = False
            elif i == "--sum" or i == "--sumup":
                sumup = True
            elif i == "--nosum" or i == "--nosumup":
                sumup = False
            elif i == "-C":
                sumup = True
                keep_coeff = True
                paramwise = False
            elif i == "-f":
                force = True
            elif i == "--keep-coeff" or i == "--keep-coeffs":
                keep_coeff = True
            elif i == "--resort":
                resort = True
            elif i == "--merge-param" or i == "--merge-params":
                paramwise = False
            elif i == "--test":
                test = True
            elif i == "--nohull" or i == "--nh":
                convexhull = False
            elif i == "--hull":
                convexhull = True
            elif i == "--hull1":
                convexhull = True
                collect = 2
            elif i == "--hull2":
                convexhull = True
                collect = 3
            elif i == "--hull3":
                convexhull = True
                collect = 5
            elif i == "--hull4":
                convexhull = True
                collect = 6
            elif i == "--ep":
                collect = 4
            elif i.startswith("-e"):
                for j in i[2:].split(","):
                    eps.append(get_ep(j))
            elif i == "--np" or i == "--no-progress":
                global progress
                set_progress(False)
            elif i == "--grid":
                collect = 7
            elif i == "-r" or i == "--round":
                collect = 8
            elif i.startswith("-r"):
                rnd = int(i[2:])
            elif i == "--log":
                collect = 9
            elif i == "--append":
                log_append = True
            elif i == "--complex" or i == "-c":
                complex = True
            elif i == "--nosave":
                save_solutions = False
            elif i == "--order":
                collect = 10
            elif i == "--fusion":
                fusion = True
            elif i == "--nofusion":
                fusion = False
            elif i == "--soltolog" or i == "--stl":
                save_solutions_to_log = True
            elif i == "--contype":
                collect = 11
            elif i == "--concomp" or i == "--cc":
                connected_components = True
            elif i == "--multiple-lpfiles":
                multiple_lpfiles = True
            elif i == "--jeff":
                jeff = True
            elif i == "--runs":
                collect = 12
            elif i == "--maxruntime":
                collect = 13
            elif i == "-q":
                verbose = 0
            elif i == "--dir":
                collect = 14
            elif i == "--db":
                collect = 15
            elif i == "--complete":
                complete = True
            elif i == "--nocomp":
                comp_sol = False
            elif i == "--nolift":
                lifting = False
            elif i == "--filter":
                collect = 16
            elif i == "--stat" or i == "-s":
                do_times = True
                full_stat = True
            elif i == "--st":
                do_times = True
            elif i == "--seed":
                collect = 17
            elif i == "--bb" or i == "--bbox":
                bounding_box = True
            elif i == "--noqd":
                qdisjoint = False
            elif i == "--qd":
                qdisjoint = True
            elif i == "--remove":
                collect = 19
            elif i == "--nonewton":
                nonewton = True
            elif i == "--rat" or i == "--rational":
                mul_denom = True
            else:
                print("unrecognized parameter '{}'".format(i))
        else:
            models.append(i)

    if test:
        sys.exit(2)

    time_prec()
    randomx.seed(random_seed)

    if not models:
        print("\n*** Please specify a model")
        sys.exit(1)

    if sumup and not keep_coeff:
        print("--sumup requires --keep-coeff !")
        sys.exit(1)

    if complete:
        common = False

    if not common:
        convexhull = False

    if not phwrap().has_v:
        convexhull = False

    if not eps:
        eps.append((fract(1, int(default_epname)), default_epname))

    if jeff:
        complex = True
        grid_data = []
        sumup = False
        keep_coeff = False
        paramwise = True
        eps = [(0, "x")]
        rnd = 0      # dummy

    if grid_data:
        save_solutions = False
        print_out = False

    if like == 15:
        import pickle
        global clf
        clf = pickle.load(open("mlmodel4.sav", "rb"))

    global mls_file
    mls_file = open("mlstats.txt", "a")

    class nonloc:
        total_runs = 0
        mismatch = 0
        matches = 0
        log_file_name = _log_file_name
        runs = _runs
    nl = nonloc()

    import traceback
    def run_it(nl):
        first_run = True
        while nl.runs > 0:
            for ep, epname in eps:
                for mod in models:
                    flags = ("s" if sumup else "") + ("k" if keep_coeff else "") + ("" if paramwise else "m")
                    pflags = flags + ("c" if complex else "")
                    sflags = pflags + ("f" if fusion else "") + ("p" if complete else "") + ("i" if filter else "")
                    mod_dir = (work_dir + os.sep if work_dir else "") + "{}{}{}{}".format(db_dir, os.sep, mod, os.sep)
                    fname0 = "{}ep{}{}".format(mod_dir, epname, ("-r{}".format(rnd) if rnd > 0 else ""))
                    tfname1 = fname0 + ("-" + flags if flags else "")
                    fname1 = fname0 + ("-" + pflags if pflags else "")
                    sfname1 = fname0 + ("-" + sflags if sflags else "")

                    if not multi_log:
                        # each model has its own log file
                        prt.close_log_file()
                        log_file_name = sfname1 + "-log.txt"

                    assert log_file_name
                    if not prt.get_log_file():
                        # log file set (through --log as multi_log or individually), but not yet opened
                        prt.set_log_file(open(log_file_name, "a" if log_append else "w"))
                        if prt.get_log_file():
                            status(boot_time, file=prt.get_log_file())

                    prt()
                    prt("-" * 70)
                    if ep == 0:
                        epstr = "x"
                    elif 1 / ep < 11000:
                        epstr = str(ep)
                    else:
                        lg = int(floor(log10(1/ep)))
                        add = 1/ep - 10**lg
                        epstr = "1/(10**{}+{})".format(lg, add) if add else "1/10**{}".format(lg)
                    tropstr = "" if jeff else "ep={}, round={}, complex={}, sumup={}, keep_coeff={}, paramwise={}, ".format(
                    epstr, rnd, complex, sumup, keep_coeff, paramwise)
                    prt("Solving {} with {}complete={}, filter={}, fusion={}, algorithm {}, resort={}, bbox={}, common={}, qdisjoint={}, nonewton={}, mul_denom={}, convexhull={}, likeness {}, backend {}".format(
                        mod, tropstr, complete, filter, fusion, algo, resort, bounding_box, common, qdisjoint, nonewton, mul_denom, (convexhull, chull_f1, chull_f2, chull_f3, chull_f4), like, phwrap().name))
                    fep = float(1/ep) ** (1/10**rnd)
                    prt("Random seed: {}".format("{}".format(np.base_repr(random_seed, 36))))
                    prt("Effective epsilon: 1/{}{}".format(fep, ", 1/(1 + {:.3g})".format(fep - 1) if fep <= 1.1 else ""))
                    prt()

                    trop_cache = None
                    ts = None
                    for override in sample_grid(grid_data):
                        # if grid sampling, print which parameters we're using
                        if grid_data:
                            param_str = ", ".join(["{} = {}".format(k, v) for k,v in override])
                            prt("\nGrid point: {}".format(param_str))

                        # load or compute tropical system
                        same_trop = False
                        if jeff:
                            ts = load_jeff_system(mod_dir)
                        else:
                            old_ts = ts
                            ts = None if force or grid_data else load_tropical(tfname1)
                            if ts is None:
                                prt("Computing tropical system... ", flush=True)
                                ts, trop_cache = tropicalize_system(mod, mod_dir, ep=float(ep), scale=10**rnd, sumup=sumup, verbose=verbose, keep_coeff=keep_coeff, paramwise=paramwise, param_override=dict(override), cache=trop_cache, mul_denom=mul_denom)
                                if ts is None:
                                    return
                                if not grid_data:
                                    save_tropical(ts, tfname1)
                                same_trop = ts == old_ts    # set to True if grid sampling and nothing changed
                        if len(ts) == 0:
                            prt("Zero-dimensional system, nothing to compute.")
                            return
                        if same_trop:
                            prt("Skipping calculation, since nothing changed.")
                            rs = old_rs
                            stats = old_stats
                            sol = None
                        else:
                            # load solution, if existant
                            sol = None
                            if comp_sol and not complex and not fusion and not grid_data and ep == fract(1, 5) and rnd == 0:
                                sol = load_known_solution(mod)

                            pts_bags = None if force or grid_data else load_polyhedra(fname1)
                            if pts_bags is None:
                                prt(end="Creating polyhedra... ", flush=True)
                                pts_bags = make_polyhedra(ts, skip_formula, verbose=verbose, complex=complex, nonewton=nonewton)
                                if not grid_data:
                                    save_polyhedra(pts_bags, fname1)
                            prt()

                            # remove all polyhedra listed through --remove option
                            phnames = []
                            for b in pts_bags:
                                delidx = []
                                for cnt,p in enumerate(b):
                                    phname = "{}.{}".format(b.name[1:], p.oidx)
                                    if phname in remove_ph:
                                        delidx.append(cnt)
                                        phnames.append(phname)
                                for i in reversed(delidx):
                                    del b[i]
                            if phnames:
                                prt("Removed polyhedra: {}\n  ({} polyhedra)".format(", ".join(phnames), len(phnames)))

                            #for b in pts_bags:
                            #    preprocess_bag(b)
                            if sorting == 1:
                                prt("Sorting ascending...")
                                pts_bags = sorted(pts_bags, key=lambda x: len(x))
                            elif sorting == 2:
                                prt("Sorting descending...")
                                pts_bags = sorted(pts_bags, key=lambda x: -len(x))
                            elif sorting == 3:
                                prt("Shuffling...")
                                randomx.shuffle(pts_bags)

                            # if --order was specified, order listed bags to the beginning
                            if bag_order:
                                bags = []
                                for i in bag_order:
                                    for j in range(len(pts_bags)):
                                        if not pts_bags[j] is None and pts_bags[j].name == "#{}".format(i):
                                            bags.append(pts_bags[j])
                                            pts_bags[j] = None
                                for b in pts_bags:
                                    if not b is None:
                                        bags.append(b)
                                pts_bags = bags
                                #sorting = 0

                            rs, stats = intersect_all(pts_bags, verbose=1+verbose, algo=algo, like=like, common=common, sorting=sorting,
                                resort=resort, convexhull=convexhull, fname1=fname1, max_run_time=max_run_time, bag_order=bag_order,
                                complete=complete, filter=filter, bbox=bounding_box)
                            old_rs, old_stats = rs, stats

                        if stats["aborted"] == 2:
                            return
                        if not stats["aborted"]:
                            #if fusion and not same_trop:
                            #    # fuse polyhedra
                            #    from fuse import fuse_polyhedra
                            #    ftime = mytime()
                            #    olen = len(rs)
                            #    fuse_polyhedra(rs, progress=True)
                            #    ftime = mytime() - ftime
                            #    if len(rs) != olen:
                            #        prt("Fusion reduced polyhedra from {} to {}.  Fusion time: {:.3f} sec".format(olen, len(rs), ftime))

                            # print number of solutions
                            s = "Solutions: {}".format(len(rs))
                            if rs:
                                s += ", dim={:.1f}, hs={:.1f}{}".format(avg_dim(rs), avg_planes(rs),
                                ", vs={:.1f}".format(avg_vrep(rs)) if phwrap().has_v else "")
                            s += ", max={}, maxin={}".format(stats["max_pt"], stats["maxin_pt"])
                            prt(s)
                            import collections
                            fvec = collections.Counter([p.dim() for p in rs])
                            if fvec:
                                prt("f-vector: {}".format([fvec.get(i, 0) for i in range(max(fvec.keys())+1)]))

                            # sort into a distict representation
                            if save_solutions or save_solutions_to_log:
                                prt("Canonicalizing...", flush=True)
                                rs = canonicalize(rs)

                            # count connected components
                            if connected_components:
                                import graph
                                adj, adj_str = graph.build_graph(rs, con_type=con_type, dbg=True)
                                cc = graph.connected_components(adj)
                                prt("Connected components: {}".format(cc))
                                #prt("Graph ID: {}".format(graph.graph_name(adj_str)))

                            prt.flush_all()

                            if (save_solutions or save_solutions_to_log or print_out) and first_run:
                                prt(end="Saving solutions... ", flush=True)
                                dim = len(rs[0].Hrep()[0].vector()) - 1 if rs else -1
                                vars = ["x{}".format(i+1) for i in range(dim)]
                                if not (save_solutions and multiple_lpfiles):
                                    s = sol_to_string(rs, vars, ep=ep if lifting else None, scale=10**rnd)
                                if save_solutions:
                                    if multiple_lpfiles:
                                        sol_to_lpfile(rs, fname1, vars, ep=ep if lifting else None, scale=10**rnd)
                                    else:
                                        sol_string_to_one_lpfile(s, sfname1)
                                if save_solutions_to_log or print_out:
                                    prt(end=s, log=save_solutions_to_log, screen=print_out)
                                prt(flush=True)

                            if sol is not None:
                                notfound = compare_solutions(rs, sol)
                                if notfound == 0:
                                    prt("Solutions match known solutions.")
                                    nl.matches += 1
                                else:
                                    prt("Solutions have {} differences from known solutions!".format(notfound))
                                    nl.mismatch += 1

                            prt("Calculation done.")
                        nl.total_runs += 1

            nl.runs -= 1
            first_run = False

    try:
        run_it(nl)
    except Exception as e:
        msg = str(e.message if hasattr(e, 'message') else e)
        prt("\n*** Program error! ***")
        prt(traceback.format_exc(), screen=False, flush=True)
        raise

    total_time = mytime() - total_time
    if do_times:
        prt("Overall intersection time: {:.3f} sec, total inclusion time: {:.3f} sec".format(overall_isect_time, overall_incl_time))
    if nl.total_runs > 1:
        prt("Overall time: {:.3f} sec for {} runs, avg. {:.3f} sec".format(total_time, nl.total_runs, total_time/nl.total_runs), flush=True)
        prt("Mismatches: {}/{}".format(nl.mismatch, nl.matches + nl.mismatch))
    prt.flush_all()

    # print memory stats
    try:
        import psutil
        p = psutil.Process().memory_info()
        peak1 = p.peak_wset
        peak2 = p.peak_pagefile
        # pmem(rss=18202624, vms=10878976, num_page_faults=4762, peak_wset=18239488, wset=18202624,
        # peak_paged_pool=183576, paged_pool=183400, peak_nonpaged_pool=14976, nonpaged_pool=14248,
        # pagefile=10878976, peak_pagefile=10985472, private=10878976)
        prt("Peak memory: working set {:.3f} GiB, pagefile {:.3f} GiB".format(peak1 / 2**30, peak2 / 2**30))
        prt("Memory: {}".format(p), screen=False)
    except:
        pass


if __name__ == "__main__":
    if not IS_SAGE:
        import doctest
        import phwrapper
        doctest.testmod(phwrapper, verbose=False)
        import bbox
        doctest.testmod(bbox, verbose=False)
        import biomd
        doctest.testmod(biomd, verbose=False)
        import util
        doctest.testmod(util, verbose=False)
        #import fuse
        #doctest.testmod(fuse, verbose=False)
        import graph
        doctest.testmod(graph, verbose=False)
        doctest.testmod(verbose=False)

    main()
