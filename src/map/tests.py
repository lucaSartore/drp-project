from map.map import *
from random import Random
import numpy as np

def test_position():
    m = Map(Settings())

    assert np.count_nonzero(m.runners_positions) != 0
    assert np.count_nonzero(m.chasers_positions) != 0

    # test runner getter
    p = m.runner.position
    assert p.x == m.runners_positions[0,0]
    assert p.y == m.runners_positions[0,1]

    # test runner setter
    m.runner.position = Point(10,20)
    assert m.runners_positions[0,0] == 10
    assert m.runners_positions[0,1] == 20

    # test chasers getter
    p = m.chasers[2].position
    assert p.x == m.runners_positions[2,0]
    assert p.y == m.runners_positions[2,1]

    # test chasers setter
    m.chasers[2].position = Point(11,21)
    assert m.chasers_positions[2,0] == 11
    assert m.chasers_positions[2,1] == 21

    # test fake_runners getter
    p = m.fake_runners[3].position
    assert p.x == m.runners_positions[4,0]
    assert p.y == m.runners_positions[4,1]

    # test fake_runners setter
    m.fake_runners[3].position = Point(12,22)
    assert m.runners_positions[4,0] == 12
    assert m.runners_positions[4,1] == 22


