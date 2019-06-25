
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

# pure python

from __future__ import print_function, division

from prt import prt


def con(a,x,y):
    a[x,y] = 1
    a[y,x] = 1

import numpy as np
def connected_components(adj, dbg=False):
    """
    find the number of connected components.
    uses an idea from SO: https://stackoverflow.com/questions/4005206/algorithm-for-counting-connected-components-of-a-graph-in-python
    that uses the idea of disjoint sets: https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    >>> a = np.zeros((0,0), dtype=bool)
    >>> connected_components(a)
    0
    >>> a = np.array([[False, False], [False, False]])
    >>> connected_components(a)
    2
    >>> a = np.zeros((2,2))
    >>> connected_components(a)
    2
    >>> a = np.zeros((2,2))
    >>> con(a,0,1)
    >>> connected_components(a)
    1
    >>> a = np.zeros((3,3))
    >>> con(a,0,1)
    >>> connected_components(a)
    2
    >>> a = np.zeros((6,6))
    >>> con(a,0,1)
    >>> con(a,1,2)
    >>> con(a,3,4)
    >>> con(a,4,5)
    >>> connected_components(a)
    2
    >>> a = np.zeros((6,6))
    >>> con(a,0,2)
    >>> con(a,1,2)
    >>> con(a,3,5)
    >>> con(a,4,5)
    >>> connected_components(a)
    2
    >>> a = np.zeros((10,10), dtype=int)
    >>> con(a,0,1)
    >>> con(a,1,2)
    >>> con(a,2,5)
    >>> con(a,3,4)
    >>> con(a,4,7)
    >>> con(a,7,8)
    >>> con(a,5,8)
    >>> connected_components(a)
    3
    """
    # init
    assert adj.shape[0] == adj.shape[1]
    l = adj.shape[0]
    sets = [set([i]) for i in range(l)]                 # first, each node is by himself
    # walk the graph
    i = 0
    while i < len(sets):
        j = i + 1
        while j < len(sets):
            if any(adj[u,v] for u in sets[i] for v in sets[j]):
                sets[j].update(sets[i])
                sets[i].clear()
            else:
                j += 1                                            # otherwise, move on
        i += 1
    sets = [s for s in sets if s]
    if dbg:
        for i in sets:
            print(i)
    return len(sets)


def is_connected(p, q, c, typ):
    if c.is_empty():
        return False
    # there is some connection
    if typ == 0:
        return True
    if typ == 2:
        return p.dim() == q.dim()
    mindim = min(p.dim(), q.dim())
    if mindim == 0:
        return True
    return c.dim() >= mindim - 1


import itertools
def build_graph(b, con_type, dbg=False):
    """
    build adjacency matrix of a bag of polytopes.
    an entry is True, iff the two polytopes are connected (i.e. the intersection is not empty)
    """
    # init
    s = ""
    l = len(b)
    adj = np.zeros((l, l), dtype=bool)
    for i,p in enumerate(b):
        p.idx = i
    # test two nodes for connection
    for p,q in itertools.combinations(b, 2):
        c = p & q
        if is_connected(p, q, c, con_type):
            if dbg:
                prt("Connection between {} and {}, dim={}/{}/{}".format(p.idx, q.idx, p.dim(), q.dim(), c.dim()))
            adj[p.idx, q.idx] = True
            adj[q.idx, p.idx] = True
            s += "{}:{}:{}/{}/{}\n".format(p.idx, q.idx, p.dim(), q.dim(), c.dim())
    if dbg:
         for p in b:
             if not any(adj[p.idx,:]):
                 prt("No connection to {}, dim={}".format(p.idx, p.dim()))
    return adj, s


def graph_name(s):
    """
    Return a (more or less) unique name for a graph.
    Hash it and return the first 12 digits in base 36.
    The chance of collision is 1 in 36**12**0.5 ~= 2e9

    #>>> graph_name("x")
    #'14RTYH2WEZM8'
    """
    import hashlib
    return np.base_repr(int(hashlib.sha256(s).hexdigest(), 16), 36)[:12]


def main(files):
    import lp2pt
    b = lp2pt.load_ph_from_fname_list(files)
    prt("Loaded {} polyhedra".format(len(b)))
    con_type = 0
    adj, adj_str = build_graph(b, con_type=con_type, dbg=True)
    cc = connected_components(adj)
    prt("Connected components: {}".format(cc))


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import sys
    files = []
    for i in sys.argv[1:]:
        files.append(i)
    if files:
        main(files)
