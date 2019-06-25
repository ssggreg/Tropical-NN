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
    print('a')
    from sage.all_cmdline import * 
    print('a') # import sage library
    from sage.libs.ppl import Variable, Linear_Expression, Constraint_System, C_Polyhedron, Generator_System, point, line, ray
    IS_SAGE = True
    IS_PPL = True
except ImportError:
    IS_SAGE = False
    from ppl import Variable, Linear_Expression, Constraint_System, C_Polyhedron, Generator_System, point, line, ray
    IS_PPL = True


phwrap_class = None

def set_phwrap(wrap):
    global phwrap_class
    phwrap_class = wrap

def phwrap(**kwargs):
    return phwrap_class(**kwargs) if kwargs else phwrap_class


# Each vector represents an (in)equality.  The zero-th element is the inhomogenous term,
# the rest are the coefficients.  An entry equal to [1,7,3,4] represents the (in)equality 1+7x1+3x2+4x3 >= 0.


class PhWrapBase:
    """
    >>> p = PhWrapPplCPolyhedron(eqns=[(-2,0,1)], ieqs=[(3, -1, 0), (0, 1, 0)])
    >>> q = PhWrapPplCPolyhedron(eqns=[(-2,0,1)], ieqs=[(3, -1, 0), (-1, 1, 0)])
    >>> p == p
    True
    >>> p != p
    False
    >>> p == q
    False
    >>> q <= p
    True
    >>> q < p
    True
    >>> p <= q
    False
    >>> p < q
    False
    >>> p <= p
    True
    >>> p < p
    False
    >>> q >= p
    False
    >>> q > p
    False
    >>> p >= q
    True
    >>> p > q
    True
    >>> p >= p
    True
    >>> p > p
    False
    """
    def __eq__(self, other):
        return self.contains(other) and other.contains(self)

    def __ne__(self, other):
        return not self.contains(other) or not other.contains(self)

    def __le__(self, other):
        return other.contains(self)

    def __ge__(self, other):
        return self.contains(other)

    def __lt__(self, other):
        return other.contains(self) and not self.contains(other)

    def __gt__(self, other):
        return self.contains(other) and not other.contains(self)



# -------------------------------------------------------------------------------------------------
# wrapper for Polyhedron
# -------------------------------------------------------------------------------------------------

if IS_SAGE:
    class PhWrapPolyhedronBase(PhWrapBase):
        has_v = True

        def __init__(self, eqns=[], ieqs=[], vertices=[], lines=[], rays=[]):
            assert self.has_v or not (vertices or lines or rays)
            self.obj = Polyhedron(eqns=eqns, ieqs=ieqs, vertices=vertices, lines=lines, rays=rays, backend=self.backend)

        def intersect(self, other):
            self.obj = self.__and__(other)
            return self

        def __and__(self, other):
            """Intersect both polyhedra and return a new object."""
            # converting to set will remove double mention of planes
            peq = set((tuple(i) for i in self.obj.equations_list()))
            pie = set((tuple(i) for i in self.obj.inequalities_list()))
            qeq = set((tuple(i) for i in other.obj.equations_list()))
            qie = set((tuple(i) for i in other.obj.inequalities_list()))
            return self.__class__(eqns=peq | qeq, ieqs=pie | qie)

        def contains(self, other):
            # return other <= self
            return other.obj._is_subpolyhedron(self.obj)

        def is_disjoint_from(self, other):
            return self.__and__(other).is_empty()

        def print(self):
            return "eqs={}\nies={}".format(self.obj.equations_list(), self.obj.inequalities_list())

        def is_empty(self):
            return self.obj.is_empty()

        def dim(self):
            return self.obj.dim()

        def space_dim(self):
            return self.obj.ambient_dim()

        def codim(self):
            return self.obj.ambient_dim() - self.obj.dim()

        def is_compact(self):
            return self.obj.is_compact()

        def Hrep(self):
            return [HobjectPh(c) for c in self.obj.Hrepresentation()]

        def n_Hrep(self):
            return self.obj.n_Hrepresentation()

        def equalities(self):
            """Return a list of all H-objects that are equalities."""
            return [HobjectPh(c) for c in self.obj.equations()]

        def n_equalities(self):
            """Return the number H-objects that are equalities."""
            return self.obj.n_equations()

        def equalities_list(self):
            """Return a list of the vectors of all H-objects that are equalities."""
            return self.obj.equations_list()

        def equalities_tuple_list(self):
            """Return a list of tuples of the vectors of all H-objects that are equalities."""
            return [tuple(c) for c in self.obj.equations_list()]

        def inequalities(self):
            """Return a list of all H-objects that are inequalities."""
            return [HobjectPh(c) for c in self.obj.inequalities()]

        def n_inequalities(self):
            """Return the number H-objects that are inequalities."""
            return self.obj.n_inequalities()

        def inequalities_list(self):
            """Return a list of the vectors of all H-objects that are inequalities."""
            return self.obj.inequalities_list()

        def inequalities_tuple_list(self):
            """Return a list of tuples of the vectors of all H-objects that are inequalities."""
            return [tuple(c) for c in self.obj.inequalities_list()]

        # V-object functions

        def center(self):
            return self.obj.center()

        def volume(self):
            return self.obj.affine_hull().volume()

        def Vrep(self):
            return self.obj.Vrepresentation()

        def n_Vrep(self):
            return self.obj.n_Vrepresentation()

        def vertices(self):
            return self.obj.vertices()

        def n_vertices(self):
            return self.obj.n_vertices()

        def vertices_list(self):
            return self.obj.vertices_list()

        def vertices_tuple_list(self):
            return [tuple(v) for v in self.obj.vertices_list()]

        def lines(self):
            return self.obj.lines()

        def lines_list(self):
            return self.obj.lines_list()

        def lines_tuple_list(self):
            return [tuple(v) for v in self.obj.lines_list()]

        def rays(self):
            return self.obj.rays()

        def rays_list(self):
            return self.obj.rays_list()

        def rays_tuple_list(self):
            return [tuple(v) for v in self.obj.rays_list()]


    class PhWrapPolyhedronPPL(PhWrapPolyhedronBase):
        """
        >>> p = PhWrapPolyhedronPPL(eqns=[(-2,0,1)], ieqs=[(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)])
        >>> p.is_empty()
        False
        >>> p.dim()
        1
        >>> p.is_compact()
        True
        >>> p.n_Hrep()
        3
        >>> [c.vector() for c in p.Hrep()]
        [(-2, 0, 1), (3, -1, 0), (0, 1, 0)]
        >>> [c.vector() for c in p.equalities()]
        [(-2, 0, 1)]
        >>> p.n_equalities()
        1
        >>> p.equalities_list()
        [[-2, 0, 1]]
        >>> p.equalities_tuple_list()
        [(-2, 0, 1)]
        >>> [c.vector() for c in p.inequalities()]
        [(3, -1, 0), (0, 1, 0)]
        >>> p.n_inequalities()
        2
        >>> p.inequalities_list()
        [[3, -1, 0], [0, 1, 0]]
        >>> p.inequalities_tuple_list()
        [(3, -1, 0), (0, 1, 0)]

        >>> p = PhWrapPolyhedronPPL(eqns=[(-2,0,1)])
        >>> q = PhWrapPolyhedronPPL(ieqs=[(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)])
        >>> c = p & q
        >>> [e.vector() for e in c.Hrep()]
        [(-2, 0, 1), (3, -1, 0), (0, 1, 0)]
        """
        name = "Polyhedron(PPL)"
        backend = "ppl"

    class PhWrapPolyhedronCDD(PhWrapPolyhedronBase):
        name = "Polyhedron(CDD)"
        backend="cdd"

    class PhWrapPolyhedronField(PhWrapPolyhedronBase):
        name = "Polyhedron(field)"
        backend="field"

    class PhWrapPolyhedronNormaliz(PhWrapPolyhedronBase):
        name = "Polyhedron(Normaliz)"
        backend = "normaliz"

    class PhWrapPolyhedronPolymake(PhWrapPolyhedronBase):
        name = "Polyhedron(Polymake)"
        backend="polymake"


    class HobjectPh:
        """
        A single H-object, i.e. an (in)equality returned by PhWrapPolyhedronXxx.
        It's basically an Equation or Inequality.

        >>> p = Polyhedron(eqns=[], ieqs=[(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)])
        >>> hs = [HobjectPh(c) for c in p.Hrepresentation()]
        >>> [c.vector() for c in hs]
        [(3, -1, 0), (3, 0, -1), (0, 1, 0), (0, 0, 1)]
        >>> [c.coeffs() for c in hs]
        [(-1, 0), (0, -1), (1, 0), (0, 1)]
        >>> [c.is_equality() for c in hs]
        [False, False, False, False]
        >>> p = Polyhedron(eqns=[(-2,0,1)], ieqs=[(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)])
        >>> hs = [HobjectPh(c) for c in p.Hrepresentation()]
        >>> [c.vector() for c in hs]
        [(-2, 0, 1), (3, -1, 0), (0, 1, 0)]
        >>> [c.is_equality() for c in hs]
        [True, False, False]
        >>> [c.is_inequality() for c in hs]
        [False, True, True]
        """
        def __init__(self, eq):
            self.eq = eq

        def vector(self):
            """Return a tuple with the inhomogenous term and the coefficients."""
            return self.eq.vector()

        def coeffs(self):
            """Return a tuple of coefficients with the inhomogenous term."""
            return self.eq.vector()[1:]

        def is_equality(self):
            return self.eq.is_equation()

        def is_inequality(self):
            return self.eq.is_inequality()

    phwrap_class = PhWrapPolyhedronPPL



# -------------------------------------------------------------------------------------------------
# wrapper for PPL_C_Polyhedron
#
# http://doc.sagemath.org/html/en/reference/libs/sage/libs/ppl.html
# http://pythonhosted.org/pplpy/
# -------------------------------------------------------------------------------------------------

if IS_PPL:
    def to_int(l):
        x = [None if i is None else int(i) for i in l]
        return tuple(x) if type(l) == tuple else x

    if IS_SAGE:
        def _myqdiv(a, b):
            return Rational((a, b))
        myqdiv = _myqdiv

        def mpz_divisor(lq):
            """
            >>> mpz_divisor((Rational((1,2)),1))
            ([1, 2], 2)
            >>> mpz_divisor((0,1))
            ([0, 1], 1)
            """
            # turn into list of mpq
            lq = [Rational(i) for i in lq]
            # calculate lcm
            d = 1
            for q in lq:
                d = lcm(d, q.denominator())
            # calculate list of numerators
            r = [(q * d).numerator() for q in lq]
            return r, d

    else:
        from gmpy2 import qdiv, mpq, lcm, mpz
        # we need at least gmpy2 2.1.0a5, because of bug in qdiv()
        if type(qdiv(8,2)) == mpz:
            # bugfree version
            myqdiv = qdiv
        else:
            # workaround for buggy version
            def _myqdiv(a, b):
                x = qdiv(a, b)
                return qdiv(x.numerator, x.denominator)
            myqdiv = _myqdiv
        assert type(myqdiv(8,2)) == mpz

        def mpz_divisor(lq):
            """
            >>> mpz_divisor((mpq(1,2),1))
            ([mpz(1), mpz(2)], mpz(2))
            >>> mpz_divisor((0,1))
            ([mpz(0), mpz(1)], mpz(1))
            """
            # turn into list of mpq
            lq = [mpq(i) for i in lq]
            # calculate lcm
            d = 1
            for q in lq:
                d = lcm(d, q.denominator)
            # calculate list of numerators
            r = [(q * d).numerator for q in lq]
            return r, d

    def make_coefficients(coeffs, div):
        """
        >>> from gmpy2 import mpz
        >>> make_coefficients((mpz(-8), mpz(-7), mpz(-8), mpz(0), mpz(-1), mpz(-4), mpz(1), mpz(-5), mpz(-3), mpz(-2), mpz(-6)), mpz(2))
        (mpz(-4), mpq(-7,2), mpz(-4), mpz(0), mpq(-1,2), mpz(-2), mpq(1,2), mpq(-5,2), mpq(-3,2), mpz(-1), mpz(-3))
        """
        return tuple([myqdiv(c, div) for c in coeffs])

    class PhWrapPplCPolyhedron(PhWrapBase):
        """
        Keep the minimized Constraint_system as self.cs.
        C_Polyhedron only supports long integers, not rationals.
        Keep *all* equalities and inequalities that led to the polyhedron as eqns_set resp. ieqs_set.

        >>> p = PhWrapPplCPolyhedron(eqns=[(-2,0,1)], ieqs=[(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)])
        >>> p.is_empty()
        False
        >>> p.dim()
        1
        >>> p.is_compact()
        True
        >>> p.n_Hrep()
        3
        >>> [c.vector_i() for c in p.Hrep()]
        [(-2, 0, 1), (3, -1, 0), (0, 1, 0)]
        >>> [c.vector_i() for c in p.equalities()]
        [(-2, 0, 1)]
        >>> p.n_equalities()
        1
        >>> p.equalities_list_i()
        [(-2, 0, 1)]
        >>> p.equalities_tuple_list_i()
        [(-2, 0, 1)]
        >>> [c.vector_i() for c in p.inequalities()]
        [(3, -1, 0), (0, 1, 0)]
        >>> p.n_inequalities()
        2
        >>> p.inequalities_list_i()
        [(3, -1, 0), (0, 1, 0)]
        >>> p.inequalities_tuple_list_i()
        [(3, -1, 0), (0, 1, 0)]

        >>> x0 = Variable(0)
        >>> x1 = Variable(1)
        >>> cs = Constraint_System()
        >>> cs.insert(x0 >= 0)
        >>> cs.insert(x0 <= 3)
        >>> cs.insert(x1 >= 0)
        >>> cs.insert(x1 <= 3)
        >>> p = PhWrapPplCPolyhedron(cs=cs)
        >>> q = p.copy()
        >>> p.cs.insert(x1 <= 2)
        >>> [c.vector_i() for c in p.Hrep()]
        [(3, -1, 0), (3, 0, -1), (0, 1, 0), (0, 0, 1), (2, 0, -1)]
        >>> [c.vector_i() for c in q.Hrep()]
        [(3, -1, 0), (3, 0, -1), (0, 1, 0), (0, 0, 1)]

        >>> p = PhWrapPplCPolyhedron(eqns=[(-2,0,1)])
        >>> q = PhWrapPplCPolyhedron(ieqs=[(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)])
        >>> c = p & q
        >>> [e.vector_i() for e in c.Hrep()]
        [(-2, 0, 1), (3, -1, 0), (0, 1, 0)]

        >>> p = PhWrapPplCPolyhedron(vertices=[(0,0),(1,0),(1,1),(0,1)])
        >>> p.inequalities_list_i()
        [(0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1)]
        >>> p.get_bounding_box_i()
        ((0, 0), (1, 1))

        >>> p = PhWrapPplCPolyhedron(ieqs=[(0,-2,1),(0,1,-2)])
        >>> p.inequalities_list_i()
        [(0, -2, 1), (0, 1, -2)]
        >>> p.equalities_list_i()
        []
        >>> p.make_v()
        >>> p.vertices()
        [(mpz(0), mpz(0))]
        >>> p.lines_i()
        []
        >>> p.rays_i()
        [(-1, -2), (-2, -1)]

        >>> # interior points or not maximal eucledian coords don't change the result
        >>> p = PhWrapPplCPolyhedron(vertices=[(0,0),(1,0.5),(0,1),(0.5,0.5)])
        >>> p.get_bounding_box_i()
        ((0, 0), (1, 1))
        >>> p = PhWrapPplCPolyhedron(vertices=[(0,0),(0,1)])
        >>> p.get_bounding_box_i()
        ((0, 0), (0, 1))

        >>> # a horizontal line at height y=5
        >>> p = PhWrapPplCPolyhedron(eqns=[(5,0,-1)])
        >>> p.make_v()
        >>> p.vertices()
        [(mpz(0), mpz(5))]
        >>> p.lines_i()
        [(1, 0)]
        >>> p.rays_i()
        []
        >>> p.get_bounding_box_i()
        ((None, 5), (None, 5))
        >>> # intersect that with the right half-space, x >= 0
        >>> p &= PhWrapPplCPolyhedron(ieqs=[(0,1,0)])
        >>> p.make_v()
        >>> p.vertices()
        [(mpz(0), mpz(5))]
        >>> p.lines_i()
        []
        >>> p.rays_i()
        [(1, 0)]
        >>> p.get_bounding_box_i()
        ((0, 5), (None, 5))

        >>> p=PhWrapPplCPolyhedron(eqns=[(3,0,0,0,0,0,0,0,0,0,0,1),(1,0,0,0,0,0,0,0,0,0,1,0),(7,1,0,0,0,0,0,0,0,2,0,0),(9,1,0,0,0,0,0,0,2,0,0,0),(3,1,0,0,0,0,0,2,0,0,0,0),(2,0,0,0,0,0,1,0,0,0,0,0),(5,1,0,0,0,2,0,0,0,0,0,0),(4,1,0,0,1,0,0,0,0,0,0,0),(4,0,0,1,0,0,0,0,0,0,0,0),(-3,1,-2,0,0,0,0,0,0,0,0,0),], ieqs=[(-3,-1,0,0,0,0,0,0,0,0,0,0),(4,1,0,0,0,0,0,0,0,0,0,0),])
        >>> p.make_v()
        >>> p.vertices()
        [(mpz(-3), mpz(-3), mpz(-4), mpz(-1), mpz(-1), mpz(-2), mpz(0), mpz(-3), mpz(-2), mpz(-1), mpz(-3)), (mpz(-4), mpq(-7,2), mpz(-4), mpz(0), mpq(-1,2), mpz(-2), mpq(1,2), mpq(-5,2), mpq(-3,2), mpz(-1), mpz(-3))]
        >>> p.lines_i()
        []
        >>> p.rays_i()
        []

        >>> from gmpy2 import mpq
        >>> p=PhWrapPplCPolyhedron(vertices=[(mpq(1,2), 1)])
        >>> p.vertices()
        [(mpq(1,2), mpz(1))]
        """
        name = "PPL_C_Polyhedron"
        has_v = False

        def __init__(self, dim=None, what=None, eqns=[], ieqs=[], vertices=[], lines=[], rays=[], cs=None, obj=None, minimize=True, keep_sets=False):
            if obj:
                # build from another PhWrapPplCPolyhedron object
                assert not (vertices or lines or rays or eqns or ieqs or cs or dim)
                self.obj = C_Polyhedron(obj.cs)
                if keep_sets:
                    self.eqns_set = set(obj.eqns_set)
                    self.ieqs_set = set(obj.ieqs_set)
            elif dim:
                assert not (vertices or lines or rays or eqns or ieqs or cs or obj)
                self.obj = C_Polyhedron(dim, what)
                self.eqns_set = set()
                self.ieqs_set = set()
            elif cs:
                # build from Constraint_system
                assert not (vertices or lines or rays or eqns or ieqs or obj or dim)
                self.obj = C_Polyhedron(cs)
                self.eqns_set = set()
                self.ieqs_set = set()
            elif vertices or lines or rays:
                # build from V-objects: vertices, lines, rays
                assert not (eqns or ieqs or cs or obj or dim)
                gs = Generator_System()
                for v in vertices:
                    v2, d = mpz_divisor(v)
                    gs.insert(point(Linear_Expression(v2, 0), d))
                for l in lines:
                    gs.insert(point(Linear_Expression(l, 0)))
                for r in rays:
                    gs.insert(point(Linear_Expression(r, 0)))
                self.obj = C_Polyhedron(gs)
                self.eqns_set = set()
                self.ieqs_set = set()
                self.make_v()
            else:
                # build from planes
                assert not (vertices or lines or rays or cs or obj or dim)
                assert eqns or ieqs
                self.cs = Constraint_System()
                for e in eqns:
                    e2, d = mpz_divisor(e)
                    self.cs.insert(Linear_Expression(e2[1:], e2[0]) == 0)
                for e in ieqs:
                    e2, d = mpz_divisor(e)
                    self.cs.insert(Linear_Expression(e2[1:], e2[0]) >= 0)
                self.obj = C_Polyhedron(self.cs)
                if keep_sets:
                    self.eqns_set = set([tuple(i) for i in eqns])
                    self.ieqs_set = set([tuple(i) for i in ieqs])
            self.cs = self.obj.minimized_constraints() if minimize else self.obj.constraints()
            if keep_sets:
                self.eqns_set |= set(self.equalities_list())
                self.ieqs_set |= set(self.inequalities_list())
            else:
                self.eqns_set = set(self.equalities_list())
                self.ieqs_set = set(self.inequalities_list())
            self.cached_dim = None

        def copy(self, minimize=False):
            """Return a (deep) copy of this object."""
            return self.__class__(obj=self, minimize=minimize)

        def intersect(self, other, keep_sets=False):
            """Intersect self with other polyhedron and modify self."""
            self.obj.intersection_assign(other.obj)
            self.cs = self.obj.minimized_constraints()
            if keep_sets:
                self.eqns_set |= other.eqns_set | set(self.equalities_tuple_list())
                self.ieqs_set |= other.ieqs_set | set(self.inequalities_tuple_list())
            else:
                self.eqns_set = set(self.equalities_tuple_list())
                self.ieqs_set = set(self.inequalities_tuple_list())
            self.gs = None           # delete v-objects
            self.cached_dim = None
            return self

        # the &= operator
        __iand__ = intersect

        def __and__(self, other):
            """Intersect self with other polyhedron and return new object."""
            newobj = self.copy()
            return newobj.intersect(other)

        def contains(self, other):
            """Return other <= self."""
            return self.obj.contains(other.obj)

        #def interior_contains(self, point):
        #    """return True if point is an interior point of self"""
        #    return ...

        def is_disjoint_from(self, other):
            return self.obj.is_disjoint_from(other.obj)

        def print(self):
            return "{}".format(self.cs)

        def is_empty(self):
            return self.dim() == -1

        def dim(self):
            if self.cached_dim is None:
                self.cached_dim = -1 if self.obj.is_empty() else self.obj.affine_dimension()
            return self.cached_dim

        def space_dim(self):
            return self.obj.space_dimension()

        def codim(self):
            return self.obj.space_dimension() - self.obj.affine_dimension()

        def is_compact(self):
            return self.obj.is_bounded()

        def Hrep(self):
            return [HobjectPPL(c) for c in self.cs]

        def n_Hrep(self):
            return len(self.cs)

        def equalities(self):
            """Return a list of all H-objects that are equalities."""
            return [HobjectPPL(c) for c in self.cs if c.is_equality()]

        def n_equalities(self):
            """Return the number of equalities."""
            return sum(1 for c in self.cs if c.is_equality())

        def equalities_list(self):
            """Return a list of vectors of all equalities."""
            return [(c.inhomogeneous_term(),) + c.coefficients() for c in self.cs if c.is_equality()]

        if IS_SAGE:
            equalities_list_i = equalities_list
        else:
            def equalities_list_i(self):
                """Return a list of vectors of all equalities."""
                return [to_int(c) for c in self.equalities_list()]

        equalities_tuple_list = equalities_list
        """Return a list of tuples of the vectors of all equalities."""

        equalities_tuple_list_i = equalities_list_i
        """Return a list of tuples of the vectors of all equalities."""

        def inequalities(self):
            """Return a list of all H-objects that are inequalities."""
            return [HobjectPPL(c) for c in self.cs if c.is_inequality()]

        def n_inequalities(self):
            """Return the number of inequalities."""
            return sum(1 for c in self.cs if c.is_inequality())

        def inequalities_list(self):
            """Return a list of vectors of all inequalities."""
            return [(c.inhomogeneous_term(),) + c.coefficients() for c in self.cs if c.is_inequality()]

        if IS_SAGE:
            inequalities_list_i = inequalities_list
        else:
            def inequalities_list_i(self):
                """Return a list of vectors of all inequalities."""
                return [to_int(c) for c in self.inequalities_list()]

        inequalities_tuple_list = inequalities_list
        """Return a list of tuples of the vectors of all inequalities."""

        inequalities_tuple_list_i = inequalities_list_i
        """Return a list of tuples of the vectors of all inequalities."""

        # V-object functions

        def make_v(self):
            self.gs = self.obj.minimized_generators()
            self.vs = []
            self.ls = []
            self.rs = []
            for v in self.gs:
                if v.is_point():
                    self.vs.append(make_coefficients(v.coefficients(), v.divisor()))
                elif v.is_line():
                    self.ls.append(v.coefficients())
                else:
                    assert v.is_ray()
                    self.rs.append(v.coefficients())

        def ensure_v(self):
            if "gs" not in self.__dict__ or self.gs is None:
                self.make_v()

        def vertices(self):
            """Return a list of vertices in internal representation."""
            return self.vs

        def n_vertices(self):
            return len(self.vs)

        def lines(self):
            """Return a list of lines in internal representation."""
            return self.ls

        def lines_i(self):
            """Return a list of lines as integers."""
            return [to_int(c) for c in self.ls]

        def n_lines(self):
            return len(self.vs)

        def rays(self):
            """Return a list of rays in internal representation."""
            return self.rs

        def rays_i(self):
            """Return a list of rays as integers."""
            return [to_int(c) for c in self.rs]

        def n_rays(self):
            return len(self.rs)

        def get_bounding_box(self):
            if self.vs:
                mins = list(self.vs[0])
                maxs = list(self.vs[0])
                # go through all other vertices, record min and max
                for v in self.vs[1:]:
                    mins = [min(a,b) for a,b in zip(mins, v)]
                    maxs = [max(a,b) for a,b in zip(maxs, v)]
                # go through all lines: if something other than zero is listed, clear out that dimension
                for v in self.ls:
                    for i,c in enumerate(v):
                        if c:
                            mins[i] = None
                            maxs[i] = None
                # go through all rays: if something other than zero is listed,
                #   clear out that dimension for max (positive) or min (negative)
                for v in self.rs:
                    for i,c in enumerate(v):
                        if c > 0:
                            maxs[i] = None
                        elif c < 0:
                            mins[i] = None
            else:
                assert not (self.ls or self.rs)
                mins = [None] * self.space_dim()
                maxs = [None] * self.space_dim()
            mins, maxs = tuple(mins), tuple(maxs)
            import bbox
            if __debug__:
                if not self <= bbox.bbox_to_polyhedron(mins, maxs):
                    print("vs=", self.vs)
                    print("ls=", self.ls)
                    print("rs=", self.rs)
                    print(self.Hrep())
                    print(mins, maxs)
            assert self <= bbox.bbox_to_polyhedron(mins, maxs)
            return mins, maxs

        def get_bounding_box_i(self):
            mins, maxs = self.get_bounding_box()
            return to_int(mins), to_int(maxs)


    class HobjectPPL:
        """
        A single H-object, i.e. an (in)equality returned by PhWrapPplCPolyhedron.
        It's basically a Constraint.

        >>> x0 = Variable(0)
        >>> x1 = Variable(1)
        >>> cs = Constraint_System()
        >>> cs.insert(x0 >= 0)
        >>> cs.insert(x0 <= 3)
        >>> cs.insert(x1 >= 0)
        >>> cs.insert(x1 <= 3)
        >>> hs = [HobjectPPL(c) for c in cs]
        >>> [c.vector_i() for c in hs]
        [(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1)]
        >>> [c.coeffs_i() for c in hs]
        [(1, 0), (-1, 0), (0, 1), (0, -1)]
        >>> [c.is_equality() for c in hs]
        [False, False, False, False]
        >>> cs.insert(x1 == 2)
        >>> hs = [HobjectPPL(c) for c in cs]
        >>> [c.vector_i() for c in hs]
        [(0, 1, 0), (3, -1, 0), (0, 0, 1), (3, 0, -1), (-2, 0, 1)]
        >>> [c.is_equality() for c in hs]
        [False, False, False, False, True]
        >>> [c.is_inequality() for c in hs]
        [True, True, True, True, False]
        """
        def __init__(self, eq):
            self.eq = eq

        def vector(self):
            """Return a tuple with the inhomogenous term and the coefficients."""
            return (self.eq.inhomogeneous_term(),) + self.eq.coefficients()

        if IS_SAGE:
            vector_i = vector
        else:
            def vector_i(self):
                """Return a tuple with the inhomogenous term and the coefficients."""
                return to_int(self.vector())

        def coeffs(self):
            """Return a tuple of coefficients with the inhomogenous term."""
            return self.eq.coefficients()

        if IS_SAGE:
            coeffs_i = coeffs
        else:
            def coeffs_i(self):
                """Return a tuple of coefficients with the inhomogenous term."""
                return to_int(self.coeffs())

        def is_equality(self):
            return self.eq.is_equality()

        def is_inequality(self):
            return self.eq.is_inequality()

        def __repr__(self):
            return str(self.eq)

    phwrap_class = PhWrapPplCPolyhedron


if __name__ == "__main__":
    import doctest
    doctest.testmod()
