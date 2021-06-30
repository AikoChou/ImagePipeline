from os import environ
from subprocess import Popen, PIPE
from enum import Enum

class NodeLabel(Enum):
    CPU = ""
    GPU = "GPU"

def set_env_variable(key, value):
    bash_variable = value
    capture = Popen(f"echo {bash_variable}", stdout=PIPE, shell=True)
    std_out, std_err = capture.communicate()
    return_code = capture.returncode

    if return_code == 0:
        evaluated_env = std_out.decode().strip()
        environ[key] = evaluated_env
    else:
        print(f"Error: Unable to find environment variable {bash_variable}")
