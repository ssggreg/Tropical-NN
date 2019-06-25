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

from phwrapper import *


#def none_min(l):
#    l = filter(lambda x: x is not None, l)
#    return min(l) if l else None

#def none_max(l):
#    l = filter(lambda x: x is not None, l)
#    return max(l) if l else None

def bbox_union(l):
    """
    Return union of a list of bounding boxes.

    >>> bbox_union([((0,0), (1,1)), ((0,0), (2,2))])
    ((0, 0), (2, 2))
    >>> bbox_union([((0,0), (1,1)), ((0,0), (2,None))])
    ((0, 0), (2, None))
    >>> bbox_union([((0,0), (1,1)), ((None,0), (2,2))])
    ((None, 0), (2, 2))
    """
    assert l
    #r = []
    #dim = len(l[0][0])
    #for i in range(dim):
    #    lmin = none_min([min[i] for min,max in l])
    #    lmax = none_max([max[i] for min,max in l])
    #    r.append((lmin, lmax))
    #return tuple(zip(*r))

    mins = None
    for mns,mxs in l:
        assert all(i is None or j is None or i <= j for i,j in zip(mns, mxs))
        if mins is None:
            mins = mns
            maxs = mxs
        else:
            mins = [None if a is None or b is None else min(a,b) for a,b in zip(mins, mns)]
            maxs = [None if a is None or b is None else max(a,b) for a,b in zip(maxs, mxs)]
    mins, maxs = tuple(mins), tuple(maxs)
    # all input boxes must lie in union
    assert all(bbox_to_polyhedron(mns, mxs) <= bbox_to_polyhedron(mins, maxs) for mns,mxs in l)
    return mins, maxs


def bbox_intersection(l):
    """
    Return intersection of a list of bounding boxes.
    >>> bbox_intersection([((0,0),(2,2)), ((1,1),(2,2))])
    ((1, 1), (2, 2))
    >>> bbox_intersection([((0,0),(2,2)), ((1,None),(2,2))])
    ((1, 0), (2, 2))
    >>> bbox_intersection([((0,0),(1,None)), ((1,1),(2,2))])
    ((1, 1), (1, 2))
    """
    assert l
    mins = None
    for mns,mxs in l:
        assert all(i is None or j is None or i <= j for i,j in zip(mns, mxs))
        if mins is None:
            mins = mns
            maxs = mxs
        else:
            mins = [None if a is None and b is None else max(filter(lambda x: x is not None, [a,b])) for a,b in zip(mins, mns)]
            maxs = [None if a is None and b is None else min(filter(lambda x: x is not None, [a,b])) for a,b in zip(maxs, mxs)]
    mins, maxs = tuple(mins), tuple(maxs)
    # intersection must lie in all input boxes
    assert all(bbox_to_polyhedron(mins, maxs) <= bbox_to_polyhedron(mns, mxs) for mns,mxs in l)
    return mins, maxs


def bbox_to_polyhedron(mins, maxs):
    """
    Convert a bounding box into a polyhedron.

    >>> p = bbox_to_polyhedron((1, 1), (2, 2))
    >>> p.make_v()
    >>> p.vertices()
    [(mpz(2), mpz(1)), (mpz(2), mpz(2)), (mpz(1), mpz(2)), (mpz(1), mpz(1))]
    >>> p.lines_i()
    []
    >>> p.rays_i()
    []

    >>> p = bbox_to_polyhedron((1, None), (2, 2))
    >>> p.make_v()
    >>> p.vertices()
    [(mpz(2), mpz(2)), (mpz(1), mpz(2))]
    >>> p.lines_i()
    []
    >>> p.rays_i()
    [(0, -1)]

    >>> p = bbox_to_polyhedron((None, None), (None, None))
    >>> p.Hrep()
    []

    >>> p = bbox_to_polyhedron((-8, -7, -8, -1, -1, -4, 0, -5, -3, -2, -6), (-3, -3, -4, 0, -1, -2, 1, -3, -2, -1, -3))
    >>> p.Hrep()
    [x4+1==0, x0+8>=0, -x2-4>=0, -x3>=0, -x0-3>=0, -x5-2>=0, -x6+1>=0, -x7-3>=0, -x8-2>=0, -x9-1>=0, -x10-3>=0, x1+7>=0, x10+6>=0, x9+2>=0, x8+3>=0, x7+5>=0, x6>=0, x5+4>=0, -x1-3>=0, x3+1>=0, x2+8>=0]

    >>> from gmpy2 import mpz, mpq
    >>> import sys
    >>> p = bbox_to_polyhedron((mpq(1,2), 1), (2, mpq(3,2)))
    >>> p.Hrep()
    [-x0+2>=0, -2*x1+3>=0, 2*x0-1>=0, x1-1>=0]
    >>> p.make_v()
    >>> p.vertices()
    [(mpz(2), mpz(1)), (mpz(2), mpq(3,2)), (mpq(1,2), mpq(3,2)), (mpq(1,2), mpz(1))]
    """
    assert len(mins) == len(maxs)
    assert mins
    assert all(i is None or j is None or i <= j for i,j in zip(mins, maxs))
    ie = []
    dim = len(mins)
    for i,c in enumerate(mins):
        if c is not None:
            ie.append([-c] + [0]*i + [1] + [0]*(dim-i-1))
    for i,c in enumerate(maxs):
        if c is not None:
            ie.append([c] + [0]*i + [-1] + [0]*(dim-i-1))
    return phwrap(ieqs=ie) if ie else phwrap(dim=dim, what="universe")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
