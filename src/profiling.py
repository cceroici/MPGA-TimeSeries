import time
import numpy as np

class GsimProfiler:
    def __init__(self):
        super(GsimProfiler, self).__init__()
        self.measurements = dict()

    def start_measurement(self, label):
        if label in list(self.measurements.keys()):
            self.measurements[label].append({"duration": None, "t_start": time.time()})
        else:
            self.measurements[label] = [{"duration": None, "t_start": time.time()}]

    def end_measurement(self, label):
        if label in list(self.measurements.keys()):
            self.measurements[label][-1] = {'duration': time.time() - self.measurements[label][-1]['t_start']}
        else:
            print("ERROR: Profiling.py:GsimProfiler():end_measurement(): Tried to end nonexistent measurement '{}'".format(label))

    def run(self, func, *args, **kwargs):
        """
        Run the provided function while measuring its execution time.
        :param func: The function to be measured.
        :param args: Arguments to be passed to the function.
        :param kwargs: Keyword arguments to be passed to the function.
        """
        label = func.__name__
        self.start_measurement(label)
        result = func(*args, **kwargs)
        self.end_measurement(label)
        return result

    def report(self, save_report_path=None):
        """
        Print a summary report of the profiled stages, including the label, average, and maximum execution time for
        each labelled stage.
        :return:
        """

        str = "Profiled Stages:\n"
        str += "{:<20s} {:<20s} {:<20s} {:<20s} {:<20s} {:<20s}\n".format("Label", "Count", "Average", "Minimum", "Maximum", "Total")
        str += "-" * 130 + "\n"

        for label, measurements in self.measurements.items():
            durations = [measurement['duration'] for measurement in measurements if measurement['duration'] is not None]

            if len(durations) > 0:
                average = sum(durations) / len(durations)
                minimum = min(durations)
                maximum = max(durations)
                count = len(durations)
                total = np.array(durations).sum()
            else:
                average = minimum = maximum = 0.0
                count = 1
                total = 0.0

            str += "{:<20s} {:<20d} {:<20.6f} {:<20.6f} {:<20.6f} {:<20.6f}\n".format(label, count, average, minimum, maximum, total)

        if save_report_path is None:
            print(str)
        else:
            with open(save_report_path, 'w') as file:
                # Write the content to the file
                file.write(str)


if __name__ == "__main__":

    profiler = GsimProfiler()

    profiler.start_measurement("step1")
    time.sleep(2)
    profiler.start_measurement("step2")
    time.sleep(1)
    profiler.end_measurement("step1")
    profiler.end_measurement("step2")
    profiler.start_measurement("step1")
    time.sleep(1)
    profiler.end_measurement("step1")

    print(profiler.measurements['step1'])
    print(profiler.measurements['step2'])
    profiler.report("temp.txt")