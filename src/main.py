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
CONTROL_INTERVALS = 2

count_d = 0
count_c = 0


while map.run():
    t1 = time()

    count_d += 1
    count_c += 1

    if count_c == CONTROL_INTERVALS:
        for controller in controllers:
            controller.control_loop(map)
            count_c = 0

    if count_d == DISPLAY_INTERVALS:
        count_d = 0
        map.draw_agents(display)
        pdf = controllers[0].get_pdf_image()
        display.update_right_side(pdf)
        display.render()
        display.clear()

    time_to_sleep = UPDATE_PERIOD - (time()-t1)
    if time_to_sleep > 0:
        sleep(time_to_sleep)

