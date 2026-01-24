from display.display import Display
from map.map import Map, Settings
from time import time, sleep


map = Map(Settings())
display = Display()

UPDATE_PERIOD = 0.01 #s
DISPLAY_INTERVALS = 5


c = 0
while map.run():
    t1 = time()

    c += 1

    if c == DISPLAY_INTERVALS:
        c = 0
        map.draw_agents(display)
        display.render()
        display.clear()

    time_to_sleep = UPDATE_PERIOD - (time()-t1)
    if time_to_sleep > 0:
        sleep(time_to_sleep)
    else:
        print("warning: time-step to slow for simulation")

