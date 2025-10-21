import logging
import cProfile
import pstats
import regex
logging.basicConfig(
   level=logging.INFO,
   format='[%(asctime)s] [%(levelname)s] %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class colors:
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def ana_profile(func):
    def wrapper(*args, **kw):
        profiler = cProfile.Profile()
        profiler.enable()
        t_res = func(*args, **kw)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('tottime')
        print(f"\n=== 性能分析结果 (按 tottime 排序) ===")
        stats.print_stats(50)
        return t_res
    return wrapper


def start_end_log(func):
    def wrapper(*args, **kw):
        n = 10
        logging.info("=" * n + "start " + func.__name__ + "=" * n)
        t_res = func(*args, **kw)
        logging.info("=" * n + "end " + func.__name__ + "=" * n)
        print('')
        return t_res
    return wrapper

