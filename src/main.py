from typing import Literal, Tuple
from chasers_logic.baseline_controller import BaselineController
from chasers_logic.gaussian_controller import GaussianController
from chasers_logic.icontroller import IController
from chasers_logic.particle_filter_controller import ParticleFilterController
from display.display import Display
from map.map import Map, Settings
from time import time, sleep
import argparse
import csv
import os

MAX_ITERATION = 2500

type test_options = Literal["baseline", "gaussian", "particle_filter"]

def get_args():
    parser = argparse.ArgumentParser(description="Run chaser simulation tests.")

    # Boolean flags
    parser.add_argument("--display", action="store_true", help="Enable visual display")
    parser.set_defaults(fix_seed=False)
    
    parser.add_argument("--no-seed", action="store_false", dest="fix_seed", 
                        help="Do not fix the random seed (default: seed is fixed)")
    parser.set_defaults(fix_seed=True)

    parser.add_argument("--no-logs", action="store_false", dest="save_logs",
                        help="Disable saving results to results.csv (default: logs are saved)")
    parser.set_defaults(save_logs=True)

    # Numeric options
    parser.add_argument("--runners", type=int, default=4, help="Number of fake runners")
    parser.add_argument("--chasers", type=int, default=3, help="Number of chasers")
    parser.add_argument("--reps", type=int, default=100, help="Number of repetitions per testcase")

    # List of test cases
    parser.add_argument(
        "--testcases", 
        nargs="+", 
        choices=["baseline", "gaussian", "particle_filter"],
        default=["baseline", "gaussian", "particle_filter"],
        help="Space-separated list of test cases to run"
    )

    return parser.parse_args()

def main():
    args = get_args()

    # Initialize CSV file if logging is enabled
    csv_file = None
    csv_writer = None
    if args.save_logs:
        file_exists = os.path.exists("results.csv")
        csv_file = open("results.csv", "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["controller", "iteration", "time_to_catch", "time_to_first_contact"])
        if not file_exists:
            csv_writer.writeheader()

    # The arguments are now accessible via args.<name>
    for o in args.testcases:
        for i in range(args.reps):
            settings = Settings(
                n_fake_runners = args.runners,
                n_chasers = args.chasers
            )
            
            if args.fix_seed:
                settings.random_seed = i
            
            # Use the args.display boolean here
            time_to_catch, time_to_first_contact = run_test(o, args.display, settings)
            
            print(f"controller={o} iteration={i} time_to_catch={time_to_catch} "
                  f"time_to_first_contact={time_to_first_contact}")
            
            # Save to CSV if logging is enabled
            if args.save_logs and csv_writer:
                assert csv_file != None
                csv_writer.writerow({
                    "controller": o,
                    "iteration": i,
                    "time_to_catch": time_to_catch,
                    "time_to_first_contact": time_to_first_contact
                })
                csv_file.flush()

    # Close CSV file if it was opened
    if csv_file:
        csv_file.close()

def run_test(
    controller: test_options,
    enable_display: bool = True,
    settings: Settings = Settings()
) -> Tuple[int, int]:

    UPDATE_PERIOD = 0.01 #s
    DISPLAY_INTERVALS = 5
    CONTROL_INTERVALS = 2

    map = Map(settings)
    if enable_display:
        display = Display(settings)
    else:
        display = None
    controller_class: type [IController]
    if controller == "baseline":
        controller_class = BaselineController
    elif controller == "gaussian":
        controller_class = GaussianController
    elif controller == "particle_filter":
        controller_class = ParticleFilterController
    else:
        raise Exception(f"invalid argument: {controller}")

    controllers: list[IController] = [
        controller_class.build(settings.n_chasers, i, settings)
        for i in range(settings.n_chasers)
    ]

    controller_class.subscribe_to_each_other(controllers)
    controller_class.start_threads(controllers)

    counter = 0
    first_contact = 0
    while map.run():
        t1 = time()

        if map.runner_visible() and first_contact == 0:
            first_contact = counter

        counter += 1

        # limit episode length
        if counter == 2500:
            if first_contact == 0:
                first_contact = 2500
            break

        if counter % CONTROL_INTERVALS == 0:
            for c in controllers:
                c.control_loop(map)

        if counter % DISPLAY_INTERVALS and enable_display:
            assert display != None
            map.draw_agents(display)
            c0 = controllers[0]
            pdf = c0.get_pdf_image()
            if pdf is not None:
                display.update_right_side(pdf)
            display.render()
            display.clear()

        time_to_sleep = UPDATE_PERIOD - (time()-t1)
        if time_to_sleep > 0 and enable_display:
            sleep(time_to_sleep)

    if enable_display:
        assert display != None
        display.close()

    return counter, first_contact


if __name__ == '__main__':
    main()
