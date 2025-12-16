import sys #standard libraries

from .main import main

if __name__ == "__main__":
    sys.exit(main())

#无论是别人直接跑 main.py，还是按题目要求跑 python -m src，都能完美运行