from chasers_logic.controller import ChaserController
from display.display import Display
from map.map import Map, Settings
from time import time, sleep

settings = Settings()
map = Map(settings)
display = Display(settings)
controllers = [
    ChaserController(settings.n_chasers, i, settings)
    for i in range(settings.n_chasers)
]

ChaserController.subscribe_to_each_other(controllers)
ChaserController.start_threads(controllers)

UPDATE_PERIOD = 0.01 #s
DISPLAY_INTERVALS = 5

c = 0


while map.run():
    t1 = time()

    for controller in controllers:
        controller.control_loop(map)

    c += 1
    if c == DISPLAY_INTERVALS:
        c = 0
        map.draw_agents(display)
        pdf = controllers[0].get_pdf_image()
        display.update_right_side(pdf)
        display.render()
        display.clear()

    time_to_sleep = UPDATE_PERIOD - (time()-t1)
    if time_to_sleep > 0:
        sleep(time_to_sleep)
    else:
        print("warning: time-step to slow for simulation")

