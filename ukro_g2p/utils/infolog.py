import atexit
from datetime import datetime


_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_run_name = None


def init(filename, run_name):
    global _file, _run_name
    _close_logfile()
    _file = open(filename, 'a')
    _file.write('\n-----------------------------------------------------------------\n')
    _file.write('Starting new training run\n')
    _file.write('-----------------------------------------------------------------\n')
    _run_name = run_name


def log(msg):
    print(msg)
    if _file is not None:
        _file.write(f'[{datetime.now().strftime(_format)[:-3]}] {msg}\n')


def _close_logfile():
  global _file
  if _file is not None:
    _file.close()
    _file = None


atexit.register(_close_logfile)
