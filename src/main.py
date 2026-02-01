from chasers_logic.messages import MeasurementMessage
from chasers_logic.pf_manager import ParticleFilterManager
from display.display import Display
from map.map import Map, Settings
from time import time, sleep
from threading import Thread
from itertools import product

settings = Settings()
map = Map(settings)
display = Display(settings)
pms = [
    ParticleFilterManager(settings.n_chasers, i, settings)
    for i in range(settings.n_chasers)
]

for (a,b) in product(pms, pms):
    if a.agent_id != b.agent_id:
        a.subscribe_to(b)

UPDATE_PERIOD = 0.01 #s
DISPLAY_INTERVALS = 5

c = 0

for i in range(settings.n_chasers):
    Thread(target= pms[i].run).start()


while map.run():
    t1 = time()


    for i in range(settings.n_chasers):
        position = map.chasers[i].position
        measure = map.detect_runner(position)
        pms[i].push_measure(MeasurementMessage(
            measure,
            position
        ))



    c += 1
    if c == DISPLAY_INTERVALS:
        c = 0
        pdf = pms[0].visualize_pdf(False)
        map.draw_agents(display)
        display.update_right_side(pdf)
        display.render()
        display.clear()

    time_to_sleep = UPDATE_PERIOD - (time()-t1)
    if time_to_sleep > 0:
        sleep(time_to_sleep)
    else:
        print("warning: time-step to slow for simulation")

